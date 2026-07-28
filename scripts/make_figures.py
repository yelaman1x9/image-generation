"""Build the README hero figure by generating the images it shows.

    python scripts/make_figures.py

    docs/demo.jpg   two blocks: distilled vs baseline at their own settings, and the
                    guidance ladder that the adherence metric cannot rank

The full contact sheets from scripts/benchmark.py are already committed under
data/reports/sheets/ and the README links them; this is the compact version that has to
carry the argument on its own above the fold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import engine                                     # noqa: E402
import scoring                                    # noqa: E402

SCALE = 2

INK = (232, 237, 243)
INK2 = (170, 178, 189)
MUTED = (125, 133, 145)
SURFACE = (16, 18, 24)
PANEL = (26, 30, 39)
GOOD = (63, 185, 80)
BAD = (248, 81, 73)
WARN = (210, 153, 34)

PROMPT = "a red fox sitting in fresh snow, wildlife photograph"
SEED = 1234
GUIDANCES = [0.0, 3.0, 7.5, 12.0]

# The guidance row is scored over the same prompt set as scripts/benchmark.py, not over the
# one prompt whose pictures are shown. On a single prompt the metric often does register
# the collapse; the finding - that it does not, on average - is a property of the set, and
# a figure that quoted one prompt would be arguing from the exception.
from benchmark import PROMPTS as BENCHMARK_PROMPTS                       # noqa: E402


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size * SCALE)


def text(draw, xy, content, fill, size, anchor="la", bold=False) -> None:
    x, y = xy[0] * SCALE, xy[1] * SCALE
    draw.text((x, y), content, fill=fill, font=font(size), anchor=anchor)
    if bold:
        draw.text((x + 0.7, y), content, fill=fill, font=font(size), anchor=anchor)


def paste(canvas, image: Image.Image, xy, side: int) -> None:
    thumb = image.convert("RGB").resize((side * SCALE, side * SCALE), Image.LANCZOS)
    canvas.paste(thumb, (xy[0] * SCALE, xy[1] * SCALE))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the README hero figure.")
    parser.add_argument("--out", default="docs/demo.jpg")
    args = parser.parse_args()

    print(f"device: {engine.device_label()}")
    print("Generating the figure's images (nothing here is reused from the benchmark):")

    # Block A - each model at the settings it is actually meant to run at.
    engine.generate(PROMPT, model="turbo", seed=SEED)              # warm up, untimed
    turbo = engine.generate(PROMPT, model="turbo", steps=1, seed=SEED)
    engine.generate(PROMPT, model="base", steps=4, seed=SEED)      # warm up the reload
    base = engine.generate(PROMPT, model="base", steps=25, seed=SEED)
    print(f"  turbo 1 step   {turbo.seconds:.2f}s")
    print(f"  base 25 steps  {base.seconds:.2f}s")

    # Block B - the guidance ladder, shown for one prompt and scored across all of them.
    ladder, ladder_scores = [], []
    for guidance in GUIDANCES:
        runs = [engine.generate(prompt, model="turbo", steps=2, guidance=guidance,
                                seed=SEED)
                for prompt in BENCHMARK_PROMPTS]
        mean = sum(scoring.adherence([r.image for r in runs], BENCHMARK_PROMPTS))
        mean /= len(runs)
        ladder.append(runs[0])                       # the fox, for the picture
        ladder_scores.append(mean)                   # all six, for the number
        print(f"  guidance {guidance:<5g} {runs[0].seconds:.2f}s   "
              f"adherence {mean:.4f} over {len(runs)} prompts")

    turbo_score, base_score = scoring.adherence([turbo.image, base.image],
                                                [PROMPT, PROMPT])

    side, gap = 196, 12
    width = 28 * 2 + side * 4 + gap * 3
    # Block B's two label lines run to about y+side+40, and the caption is placed relative
    # to that row rather than to the canvas bottom, so the two cannot drift into each other.
    height = 96 + side + 84 + side + 146

    canvas = Image.new("RGB", (width * SCALE, height * SCALE), SURFACE)
    draw = ImageDraw.Draw(canvas)

    text(draw, (28, 22), "Two dials, both usually set by habit", INK, 17, bold=True)
    text(draw, (28, 46), f"Same prompt, same seed ({SEED}), on an "
                         f"{engine.device_label().split('(')[-1].rstrip(')')}", INK2, 12)

    # ---------------------------------------------------------------- block A
    y = 80
    text(draw, (28, y), "Distilled against undistilled, each at its own correct settings",
         INK, 13, bold=True)
    y += 20
    for index, (result, score, label) in enumerate((
            (turbo, turbo_score, "SD-Turbo, 1 step"),
            (base, base_score, "SD 1.5, 25 steps"))):
        x = 28 + index * (side + gap)
        paste(canvas, result.image, (x, y), side)
        text(draw, (x, y + side + 8), label, INK, 12, bold=True)
        text(draw, (x, y + side + 26), f"{result.seconds:.2f}s   adherence {score:.4f}",
             INK2, 11)

    ratio = base.seconds / turbo.seconds
    note_x = 28 + 2 * (side + gap) + 8
    text(draw, (note_x, y + 10), f"{ratio:.0f}x the time", GOOD, 22, bold=True)
    text(draw, (note_x, y + 44), f"for {base_score - turbo_score:+.4f} adherence.",
         INK2, 13)
    # No em dashes: the bundled font has no glyph for one and draws a box instead.
    for offset, sentence in enumerate((
            "Prompt adherence is CLIP cosine between",
            "the image and the words that asked for",
            "it. It measures whether the picture shows",
            "the right thing, and nothing at all about",
            "whether it looks good - which is what",
            "the row below is about.")):
        text(draw, (note_x, y + 72 + offset * 15), sentence, MUTED, 11)

    # ---------------------------------------------------------------- block B
    y += side + 62
    text(draw, (28, y), "Guidance on the distilled model, at 2 steps", INK, 13, bold=True)
    text(draw, (28, y + 18),
         f"SD 1.5's default is 7.5, and it is the number everyone carries over. Scores are "
         f"means over {len(BENCHMARK_PROMPTS)} prompts; the pictures are one of them.",
         INK2, 12)
    y += 42

    best = max(ladder_scores)
    for index, (guidance, result, score) in enumerate(zip(GUIDANCES, ladder,
                                                          ladder_scores)):
        x = 28 + index * (side + gap)
        paste(canvas, result.image, (x, y), side)
        colour = WARN if guidance == 7.5 else INK
        text(draw, (x, y + side + 8), f"guidance {guidance:g}", colour, 12, bold=True)
        marker = "  <- scores best" if score == best else ""
        text(draw, (x, y + side + 26), f"adherence {score:.4f}{marker}",
             GOOD if score == best else INK2, 11)

    # Written from the numbers this run actually produced. An earlier version asserted the
    # metric ranks guidance 3 top - true across the six-prompt benchmark, not on every
    # single prompt - which would have been a caption its own figure contradicted.
    at_default = ladder_scores[GUIDANCES.index(7.5)]
    gap_75 = ladder_scores[0] - at_default
    models = abs(base_score - turbo_score)
    winner = GUIDANCES[ladder_scores.index(best)]
    caption_y = y + side + 56
    text(draw, (28, caption_y),
         f"Left to right the image goes from a photograph to noise. The metric barely "
         f"notices: guidance 7.5 - visibly wrecked -", BAD, 12)
    text(draw, (28, caption_y + 18),
         f"scores {at_default:.4f} against {ladder_scores[0]:.4f} for the clean run, a gap "
         f"of {gap_75:.4f}. It ranks guidance {winner:g} best of the four.", BAD, 12)
    text(draw, (28, caption_y + 36),
         "Prompt adherence cannot tune this dial. That judgement is made by looking, which "
         "is why the contact sheets are committed.", BAD, 12)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.resize((width, height), Image.LANCZOS).save(out, format="JPEG", quality=90,
                                                       optimize=True)
    print(f"\n  wrote {out}  ({width}x{height})")

    # The caption claims the metric barely separates a wrecked image from a clean one.
    # Check it against the numbers this run produced rather than trusting the wording.
    if gap_75 > 0.02:
        print(f"\n  WARNING: guidance 7.5 scored {gap_75:.4f} below guidance 0, a clear "
              f"penalty.\n  The caption says the metric barely notices. Rewrite it before "
              f"shipping.")
        return 1
    print(f"\n  verified: guidance 7.5 is visibly wrecked and scores only {gap_75:.4f} "
          f"below the clean\n            run, with guidance {winner:g} ranked best - "
          f"the caption's claim")
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
