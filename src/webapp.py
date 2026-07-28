"""HTTP API and UI.

The GPU is serialised behind one lock rather than a job queue: at turbo step counts a
request finishes in well under a second, and a polling protocol would cost more latency
than it saves. The sweep endpoints are the slow ones, and they are slow because they are
generating a dozen images on purpose.
"""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

import engine
import scoring

DATA = Path("data")
OUTPUTS = DATA / "outputs"
STATIC = Path(__file__).resolve().parent / "static"

OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="image-generation", docs_url="/api/docs", redoc_url=None)
gpu = threading.Lock()

STEP_LADDER = {"turbo": [1, 2, 4, 8], "base": [4, 8, 16, 25, 40]}
GUIDANCE_LADDER = [0.0, 1.0, 3.0, 7.5, 12.0]


def store(image: Image.Image) -> str:
    name = f"{uuid.uuid4().hex[:12]}.png"
    image.save(OUTPUTS / name, format="PNG")
    return name


def run(**kwargs) -> engine.Result:
    with gpu:
        return engine.generate(**kwargs)


def payload(result: engine.Result, prompt: str, *, with_score: bool = True) -> dict:
    body = result.as_dict()
    body["file"] = store(result.image)
    if with_score:
        with gpu:
            body["adherence"] = round(scoring.score(result.image, prompt), 4)
    return body


def read_image(upload: UploadFile | None) -> Image.Image | None:
    if upload is None:
        return None
    raw = upload.file.read()
    if not raw:
        return None
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(422, f"could not read that image ({exc})") from None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config() -> dict:
    report_file = DATA / "reports" / "benchmark.json"
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else None

    return {
        "device": engine.device_label(),
        "default_model": engine.DEFAULT_MODEL,
        "models": {key: {"label": m.label, "distilled": m.distilled, "steps": m.steps,
                         "guidance": m.guidance, "max_steps": m.max_steps,
                         "size": m.size, "note": m.note}
                   for key, m in engine.MODELS.items()},
        "step_ladder": STEP_LADDER,
        "guidance_ladder": GUIDANCE_LADDER,
        "benchmark": report,
    }


@app.post("/api/generate")
def generate(prompt: str = Form(...), model: str = Form(engine.DEFAULT_MODEL),
             steps: int | None = Form(None), guidance: float | None = Form(None),
             seed: int | None = Form(None), negative: str = Form(""),
             strength: float = Form(0.6),
             init: UploadFile | None = File(None)) -> JSONResponse:
    if not prompt.strip():
        raise HTTPException(400, "a prompt is required")
    if model not in engine.MODELS:
        raise HTTPException(400, f"unknown model {model!r}")

    result = run(prompt=prompt, model=model, steps=steps, guidance=guidance, seed=seed,
                 negative=negative, image=read_image(init), strength=strength)
    return JSONResponse(payload(result, prompt))


@app.post("/api/compare")
def compare(prompt: str = Form(...), seed: int | None = Form(None)) -> JSONResponse:
    """The same prompt and seed through both models, each at its own correct settings."""
    if not prompt.strip():
        raise HTTPException(400, "a prompt is required")

    if seed is None or seed < 0:
        import torch
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())

    entries = []
    for key in ("turbo", "base"):
        result = run(prompt=prompt, model=key, seed=seed)
        entry = payload(result, prompt)
        entry["label"] = engine.MODELS[key].label
        entries.append(entry)

    fastest = min(e["seconds"] for e in entries)
    for entry in entries:
        entry["slower_by"] = round(entry["seconds"] / fastest, 1)
    return JSONResponse({"seed": seed, "results": entries})


@app.post("/api/sweep")
def sweep(prompt: str = Form(...), model: str = Form(engine.DEFAULT_MODEL),
          axis: str = Form("steps"), seed: int | None = Form(None)) -> JSONResponse:
    """Hold everything fixed and move one dial, so the effect is attributable."""
    if not prompt.strip():
        raise HTTPException(400, "a prompt is required")
    if model not in engine.MODELS:
        raise HTTPException(400, f"unknown model {model!r}")
    if axis not in ("steps", "guidance"):
        raise HTTPException(400, "axis must be 'steps' or 'guidance'")

    if seed is None or seed < 0:
        import torch
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())

    values = STEP_LADDER[model] if axis == "steps" else GUIDANCE_LADDER
    entries = []
    for value in values:
        kwargs = {"steps": value} if axis == "steps" else {"guidance": value}
        result = run(prompt=prompt, model=model, seed=seed, **kwargs)
        entry = payload(result, prompt)
        entry["value"] = value
        entries.append(entry)

    return JSONResponse({"seed": seed, "axis": axis, "model": model, "results": entries})


@app.get("/api/outputs/{name}")
def output(name: str) -> FileResponse:
    path = (OUTPUTS / name).resolve()
    if path.parent != OUTPUTS.resolve() or not path.exists():
        raise HTTPException(404, "no such image")
    return FileResponse(path, media_type="image/png")


@app.get("/api/reports/sheets/{name}")
def sheet(name: str) -> FileResponse:
    """The benchmark contact sheets - the part of the result a table cannot carry."""
    sheets = DATA / "reports" / "sheets"
    path = (sheets / name).resolve()
    if path.parent != sheets.resolve() or not path.exists():
        raise HTTPException(404, "no such sheet")
    return FileResponse(path, media_type="image/jpeg")
