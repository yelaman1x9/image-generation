"""Diffusion pipelines, loaded lazily and kept to one on the GPU at a time.

Two models, deliberately, because the interesting result is the comparison:

    sd-turbo    adversarially distilled from SD 2.1. Produces an image in one to four
                steps, and is trained to run with classifier-free guidance switched OFF.
    SD 1.5      the undistilled baseline. Needs 20-50 steps and guidance around 7.5.

The defaults below are per-model rather than global, because the same settings are right
for one and wrong for the other by an order of magnitude. scripts/benchmark.py measures
both surfaces; `advice()` is where those measurements are wired back into the product.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass

import torch
from PIL import Image

FAST_MODEL = os.environ.get("FAST_MODEL", "stabilityai/sd-turbo")
BASE_MODEL = os.environ.get("BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    id: str
    label: str
    distilled: bool
    steps: int
    guidance: float
    max_steps: int
    size: int
    note: str


MODELS: dict[str, ModelSpec] = {
    "turbo": ModelSpec(
        key="turbo", id=FAST_MODEL, label="SD-Turbo (distilled)", distilled=True,
        steps=2, guidance=0.0, max_steps=12, size=512,
        note="Distilled for 1-4 steps with guidance off. More of either makes it worse, "
             "not better - measured in the README."),
    "base": ModelSpec(
        key="base", id=BASE_MODEL, label="Stable Diffusion 1.5", distilled=False,
        steps=25, guidance=7.5, max_steps=60, size=512,
        note="The undistilled baseline. Needs real step counts and real guidance."),
}

DEFAULT_MODEL = "turbo"

_lock = threading.Lock()
_loaded: dict[str, object] = {}          # {"key": ..., "text": pipe, "image": pipe}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype() -> torch.dtype:
    # fp16 on CPU is emulated and slower than fp32, so it is only worth it on the GPU.
    return torch.float16 if device() == "cuda" else torch.float32


def device_label() -> str:
    if device() == "cpu":
        return "cpu"
    return f"cuda ({torch.cuda.get_device_name(0)})"


def spec(key: str | None) -> ModelSpec:
    chosen = key or DEFAULT_MODEL
    if chosen not in MODELS:
        raise KeyError(f"unknown model {chosen!r}; have {sorted(MODELS)}")
    return MODELS[chosen]


def load(key: str) -> tuple:
    """Return (text2image, image2image) pipelines for one model.

    The two share every weight - `from_pipe` re-wraps the same modules with a different
    scheduler loop - so having both costs one copy of the model, not two. Only one *model*
    stays resident: on an 8 GB card, holding SD-Turbo and SD 1.5 at once leaves too little
    room for activations at 512x512, and swapping costs a few seconds against an
    out-of-memory error that costs the request.
    """
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

    with _lock:
        if _loaded.get("key") == key:
            return _loaded["text"], _loaded["image"]

        if _loaded:
            _loaded.clear()
            gc.collect()
            if device() == "cuda":
                torch.cuda.empty_cache()

        model = spec(key)
        # SAFETY_CHECKER=1 loads Stable Diffusion's NSFW classifier, which blanks flagged
        # outputs. Off by default here because it costs about 1.2 GB of an 8 GB card and
        # fires on plenty of innocuous images - a black square in place of a lighthouse
        # makes a poor demonstration. That trade is fine for a local research tool and is
        # not fine for anything public; flip it on before exposing this to other people.
        extra = {}
        if os.environ.get("SAFETY_CHECKER", "0") != "1":
            extra = {"safety_checker": None, "requires_safety_checker": False}

        text = AutoPipelineForText2Image.from_pretrained(
            model.id, torch_dtype=dtype(),
            variant="fp16" if dtype() == torch.float16 else None, **extra)
        text = text.to(device())
        text.set_progress_bar_config(disable=True)

        image = AutoPipelineForImage2Image.from_pipe(text)
        image.set_progress_bar_config(disable=True)

        _loaded.update({"key": key, "text": text, "image": image})
        return text, image


def advice(model: ModelSpec, steps: int, guidance: float) -> list[str]:
    """Warnings the measurements justify, surfaced where the settings are chosen.

    Every number here comes from scripts/benchmark.py on this hardware, and one of them
    contradicts what this function originally said. The first draft asserted that guidance
    above 1 on a distilled model always makes things worse. It does make the *image* worse -
    washed out by 3, posterised by 7.5 - but CLIP prompt adherence peaks at guidance 3 and
    is still within 0.006 of its best at 7.5, where the pictures are visibly wrecked. So the
    warning stands, and the metric that was supposed to justify it does not. Look at
    data/reports/sheets/guidance_turbo.jpg before believing any single number, this one
    included.
    """
    notes = []
    if model.distilled and guidance > 1.0:
        notes.append(
            f"Guidance {guidance:g} on a distilled model. SD-Turbo was trained without "
            f"classifier-free guidance: by 3 the image is washing out, by 7.5 it is "
            f"posterised. Prompt-adherence scores barely register this, so it is a "
            f"judgement made by eye. Use 0.")
    if model.distilled and steps > 4:
        notes.append(
            f"{steps} steps on a distilled model. Adherence measured *highest* at 1 step "
            f"and falls away steadily - by 8 the prompt starts being dropped - while the "
            f"time triples.")
    if not model.distilled and steps < 10:
        notes.append(
            f"{steps} steps without distillation is too few. SD 1.5 measured 0.285 "
            f"adherence at 4 steps against 0.335 at 16; the image will be undercooked.")
    if not model.distilled and guidance <= 1.0:
        notes.append(
            "Guidance at or below 1 disables classifier-free guidance, which SD 1.5 "
            "depends on; the prompt will barely be followed.")
    return notes


@dataclass
class Result:
    image: Image.Image
    seed: int
    steps: int
    guidance: float
    model: str
    seconds: float
    peak_vram_mb: float
    notes: list[str]

    def as_dict(self) -> dict:
        return {"seed": self.seed, "steps": self.steps, "guidance": self.guidance,
                "model": self.model, "seconds": round(self.seconds, 3),
                "peak_vram_mb": round(self.peak_vram_mb, 1), "notes": self.notes}


def _generator(seed: int | None) -> tuple:
    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    return torch.Generator(device=device()).manual_seed(seed), seed


def generate(prompt: str, *, model: str | None = None, steps: int | None = None,
             guidance: float | None = None, seed: int | None = None,
             negative: str = "", width: int | None = None,
             height: int | None = None, image: Image.Image | None = None,
             strength: float = 0.6) -> Result:
    """One image. `image` switches to image-to-image, `strength` says how far to travel."""
    model_spec = spec(model)
    text_pipe, image_pipe = load(model_spec.key)

    steps = model_spec.steps if steps is None else max(1, min(steps, model_spec.max_steps))
    guidance = model_spec.guidance if guidance is None else max(0.0, min(guidance, 20.0))
    width = width or model_spec.size
    height = height or model_spec.size
    generator, seed = _generator(seed)

    # A negative prompt is realised through classifier-free guidance; with guidance off it
    # is not merely weak but literally unused, so saying so beats silently ignoring it.
    notes = advice(model_spec, steps, guidance)
    if negative and guidance <= 1.0:
        notes.append("A negative prompt needs guidance above 1 to have any effect; with "
                     "guidance off it is ignored entirely.")

    if device() == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    if image is not None:
        # Image-to-image starts partway along the schedule, so the *effective* step count
        # is steps x strength. Diffusers rounds that down, and at turbo step counts it can
        # round to zero - which returns the input image unchanged and looks like a bug.
        effective = max(1, int(steps * strength))
        if effective * strength < 1:
            steps = max(steps, int(1 / max(strength, 0.05)) + 1)
        output = image_pipe(prompt=prompt, image=image, strength=strength,
                            num_inference_steps=steps, guidance_scale=guidance,
                            negative_prompt=negative or None, generator=generator)
    else:
        output = text_pipe(prompt=prompt, num_inference_steps=steps,
                           guidance_scale=guidance, negative_prompt=negative or None,
                           width=width, height=height, generator=generator)
    if device() == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    peak = torch.cuda.max_memory_allocated() / 1e6 if device() == "cuda" else 0.0
    return Result(output.images[0], seed, steps, guidance, model_spec.key,
                  elapsed, peak, notes)


def release() -> None:
    """Drop the resident pipeline and free the CUDA context deliberately."""
    with _lock:
        _loaded.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
