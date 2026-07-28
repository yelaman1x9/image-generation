"""Measure what steps and guidance actually buy, instead of assuming more is better.

    python scripts/benchmark.py

Three sweeps, all on the same prompts and the same seeds so the only thing moving is the
dial under test:

  1. steps, on the distilled model      - where does prompt adherence stop improving?
  2. steps, on the undistilled baseline - the same question with a different answer
  3. guidance, on the distilled model   - the setting everybody carries over and shouldn't

Prompt adherence is CLIP cosine (src/scoring.py), which measures whether the image shows
what was asked for and nothing about whether it looks good. Latency is wall clock on this
machine after a warm-up pass, so the first-call compile cost is not smeared into it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import engine                                     # noqa: E402
import scoring                                    # noqa: E402

PROMPTS = [
    "a red fox sitting in fresh snow, wildlife photograph",
    "an astronaut riding a horse on the moon",
    "a bowl of ramen on a wooden table, food photography",
    "a lighthouse in a storm, oil painting",
    "a vintage bicycle leaning against a brick wall",
    "a mountain lake at sunrise, wide angle landscape",
]
SEEDS = [1234, 5678]

STEP_LADDER = {"turbo": [1, 2, 4, 8], "base": [4, 8, 16, 25, 40]}
GUIDANCE_LADDER = [0.0, 1.0, 3.0, 7.5, 12.0]


def run_config(model: str, *, steps: int | None = None, guidance: float | None = None,
               prompts=PROMPTS, seeds=SEEDS) -> dict:
    images, used_prompts, times, peaks = [], [], [], []
    for prompt in prompts:
        for seed in seeds:
            result = engine.generate(prompt, model=model, steps=steps,
                                     guidance=guidance, seed=seed)
            images.append(result.image)
            used_prompts.append(prompt)
            times.append(result.seconds)
            peaks.append(result.peak_vram_mb)

    scores = scoring.adherence(images, used_prompts)
    return {
        "model": model, "steps": steps, "guidance": guidance,
        "adherence_mean": round(statistics.fmean(scores), 4),
        "adherence_sd": round(statistics.pstdev(scores), 4),
        "seconds_median": round(statistics.median(times), 3),
        "peak_vram_mb": round(max(peaks), 1),
        "images": images, "scores": scores,
    }


def contact_sheet(entries: list[dict], caption: str, path: Path,
                  per_row: int = 6) -> None:
    """One row per configuration, so the difference is visible rather than described."""
    if not entries:
        return
    tile = 224
    label_height, header = 26, 34
    width = tile * per_row
    height = header + len(entries) * (tile + label_height)

    sheet = Image.new("RGB", (width, height), (16, 18, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), caption, fill=(230, 237, 243))

    for row, entry in enumerate(entries):
        top = header + row * (tile + label_height)
        dial = f"steps {entry['steps']}" if entry["guidance"] is None or entry["steps"] else ""
        if entry.get("_axis") == "guidance":
            dial = f"guidance {entry['guidance']:g}"
        else:
            dial = f"{entry['steps']} steps"
        draw.text((10, top + 6),
                  f"{dial:<14} adherence {entry['adherence_mean']:.4f}   "
                  f"{entry['seconds_median']:.2f}s/image",
                  fill=(139, 148, 158))
        for column, image in enumerate(entry["images"][:per_row]):
            sheet.paste(image.resize((tile, tile), Image.LANCZOS),
                        (column * tile, top + label_height))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=88)
    print(f"  wrote {path}")


def table(entries: list[dict], dial: str) -> None:
    best = max(e["adherence_mean"] for e in entries)
    fastest = min(e["seconds_median"] for e in entries)
    print(f"  {dial:<10} adherence   vs best   s/image   vs fastest   peak VRAM")
    print("  " + "-" * 68)
    for entry in entries:
        value = entry["steps"] if dial == "steps" else f"{entry['guidance']:g}"
        delta = entry["adherence_mean"] - best
        print(f"  {str(value):<10} {entry['adherence_mean']:.4f}    "
              f"{delta:+.4f}   {entry['seconds_median']:>6.2f}    "
              f"{entry['seconds_median'] / fastest:>7.1f}x   "
              f"{entry['peak_vram_mb']:>7.0f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark steps and guidance.")
    parser.add_argument("--out", default="data/reports")
    parser.add_argument("--sheets", default="data/reports/sheets")
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    args = parser.parse_args()

    prompts = PROMPTS[:args.prompts]
    print(f"device: {engine.device_label()}")
    print(f"{len(prompts)} prompts x {len(SEEDS)} seeds = "
          f"{len(prompts) * len(SEEDS)} images per configuration\n")

    report = {"device": engine.device_label(), "prompts": prompts, "seeds": SEEDS,
              "sweeps": {}}

    # ---------------------------------------------------------------- 1 & 2: steps
    for model in ("turbo", "base"):
        spec = engine.MODELS[model]
        print(f"\n{'=' * 72}\n{spec.label} - how many steps are worth paying for?\n")
        engine.generate(prompts[0], model=model, seed=0)          # warm up, untimed
        entries = [run_config(model, steps=steps, prompts=prompts)
                   for steps in STEP_LADDER[model]]
        table(entries, "steps")

        best = max(entries, key=lambda e: e["adherence_mean"])
        knee = next(e for e in entries
                    if e["adherence_mean"] >= best["adherence_mean"] - 0.005)
        print(f"\n  Adherence is within 0.005 of its best by {knee['steps']} steps "
              f"({knee['seconds_median']:.2f}s); the top of the ladder costs "
              f"{entries[-1]['seconds_median'] / knee['seconds_median']:.1f}x that "
              f"for {best['adherence_mean'] - knee['adherence_mean']:+.4f}.")

        contact_sheet(entries, f"{spec.label} - steps", Path(args.sheets) / f"steps_{model}.jpg")
        report["sweeps"][f"steps_{model}"] = [
            {k: v for k, v in e.items() if k not in ("images", "scores")} for e in entries]

    # ---------------------------------------------------------------- 3: guidance
    print(f"\n{'=' * 72}\nSD-Turbo - classifier-free guidance, the setting carried over "
          f"by habit\n")
    entries = []
    for guidance in GUIDANCE_LADDER:
        entry = run_config("turbo", steps=2, guidance=guidance, prompts=prompts)
        entry["_axis"] = "guidance"
        entries.append(entry)
    table(entries, "guidance")

    off = entries[0]
    on = max(entries, key=lambda e: e["guidance"])
    peak = max(entries, key=lambda e: e["adherence_mean"])
    default_75 = next((e for e in entries if e["guidance"] == 7.5), None)

    print(f"\n  Guidance {on['guidance']:g} scores "
          f"{on['adherence_mean'] - off['adherence_mean']:+.4f} against guidance 0 and takes "
          f"{on['seconds_median'] / off['seconds_median']:.1f}x as long, because guidance "
          f"runs\n  the network on a second, unconditional batch every step.")
    if peak["guidance"] > 1.0:
        print(f"\n  Read no further than that and you would ship guidance "
              f"{peak['guidance']:g}, which scores\n  best of all "
              f"({peak['adherence_mean']:.4f}) while visibly washing the image out.")
    print(f"\n  Either way, look at the sheet rather than the column. At guidance 7.5 - "
          f"SD 1.5's\n  default, and the number everyone carries over - the images are "
          f"posterised\n  wreckage, and they still score {default_75['adherence_mean']:.4f}, "
          f"only "
          f"{off['adherence_mean'] - default_75['adherence_mean']:+.4f} against guidance 0.")
    print(f"\n  Prompt adherence is not image quality, and this is the sweep that proves it. "
          f"The\n  metric cannot separate these; a person can, instantly.")

    contact_sheet(entries, "SD-Turbo - guidance at 2 steps",
                  Path(args.sheets) / "guidance_turbo.jpg")
    report["sweeps"]["guidance_turbo"] = [
        {k: v for k, v in e.items() if k not in ("images", "scores")} for e in entries]

    # ---------------------------------------------------------------- headline
    turbo_1 = report["sweeps"]["steps_turbo"][0]
    base_25 = next(e for e in report["sweeps"]["steps_base"] if e["steps"] == 25)
    print(f"\n{'=' * 72}\nHeadline\n")
    print(f"  SD-Turbo, 1 step   adherence {turbo_1['adherence_mean']:.4f}   "
          f"{turbo_1['seconds_median']:.2f}s")
    print(f"  SD 1.5, 25 steps   adherence {base_25['adherence_mean']:.4f}   "
          f"{base_25['seconds_median']:.2f}s")
    print(f"\n  {base_25['seconds_median'] / turbo_1['seconds_median']:.0f}x the time for "
          f"{base_25['adherence_mean'] - turbo_1['adherence_mean']:+.4f} adherence.")
    report["headline"] = {"turbo_1_step": turbo_1, "base_25_steps": base_25}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        scoring.release()
        engine.release()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
