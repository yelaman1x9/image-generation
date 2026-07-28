"""Command-line access to the same engine the web app uses.

    python src/cli.py "a red fox in fresh snow"
    python src/cli.py "a red fox in fresh snow" --model base --steps 30
    python src/cli.py "make it winter" --init photo.jpg --strength 0.55
    python src/cli.py "a lighthouse in a storm" --sweep steps
    python src/cli.py info
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine                                     # noqa: E402
import scoring                                    # noqa: E402


def emit(result: engine.Result, prompt: str, path: Path, *, score: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(path)
    line = (f"  {path}  seed {result.seed}  {result.steps} steps  "
            f"guidance {result.guidance:g}  {result.seconds:.2f}s")
    if result.peak_vram_mb:
        line += f"  peak {result.peak_vram_mb:.0f} MB"
    if score:
        line += f"  adherence {scoring.score(result.image, prompt):.4f}"
    print(line)
    for note in result.notes:
        print(f"    note: {note}")


def cmd_generate(args) -> int:
    init = Image.open(args.init).convert("RGB") if args.init else None
    out = Path(args.out)

    if args.sweep:
        values = ({"turbo": [1, 2, 4, 8], "base": [4, 8, 16, 25, 40]}[args.model]
                  if args.sweep == "steps" else [0.0, 1.0, 3.0, 7.5, 12.0])
        print(f"\nSweeping {args.sweep} on {engine.MODELS[args.model].label}, "
              f"seed fixed at {args.seed}\n")
        for value in values:
            kwargs = {"steps": value} if args.sweep == "steps" else {"guidance": value}
            result = engine.generate(args.prompt, model=args.model, seed=args.seed,
                                     negative=args.negative, image=init,
                                     strength=args.strength, **kwargs)
            emit(result, args.prompt, out / f"{args.sweep}_{value}.png")
        return 0

    print()
    for index in range(args.count):
        seed = None if args.seed is None else args.seed + index
        result = engine.generate(args.prompt, model=args.model, steps=args.steps,
                                 guidance=args.guidance, seed=seed,
                                 negative=args.negative, image=init,
                                 strength=args.strength)
        emit(result, args.prompt, out / f"{result.seed}.png")
    return 0


def cmd_info(args) -> int:
    print(f"\nDevice : {engine.device_label()}")
    print(f"Default: {engine.DEFAULT_MODEL}\n")
    for key, spec in engine.MODELS.items():
        print(f"  {key:<7} {spec.label}")
        print(f"          {spec.id}")
        print(f"          defaults: {spec.steps} steps, guidance {spec.guidance:g}, "
              f"{spec.size}x{spec.size}")
        print(f"          {spec.note}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `info` is handled before argparse rather than as a subparser: a subcommand and a bare
    # positional prompt cannot coexist, and `cli.py "a red fox"` is the shape that matters.
    if argv and argv[0] == "info":
        return cmd_info(None)

    parser = argparse.ArgumentParser(
        description="Generate images. Use `cli.py info` for models and defaults.")
    parser.add_argument("prompt")
    parser.add_argument("--model", default=engine.DEFAULT_MODEL,
                        choices=sorted(engine.MODELS))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--negative", default="")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--init", default=None, help="image-to-image source")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--sweep", choices=["steps", "guidance"], default=None)
    parser.add_argument("--out", default="data/outputs")

    return cmd_generate(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        code = main()
    finally:
        scoring.release()
        engine.release()
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit: tearing down a CUDA context while worker threads finalise can hang or
    # segfault a run that already succeeded.
    os._exit(code)
