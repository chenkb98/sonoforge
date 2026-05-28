from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__, audio, diffusion, experiments, synthesis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sonoforge")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate a local audio experiment")
    generate.add_argument("--prompt", default="sine 440")
    generate.add_argument("--duration", type=float, default=0.1)
    generate.add_argument("--sample-rate", type=int, default=16000)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", required=True)
    train = commands.add_parser("train-demo", help="Fit a tiny synthetic denoiser baseline")
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output", required=True)
    inspect = commands.add_parser("inspect", help="Inspect a PCM16 WAV file")
    inspect.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            settings = dict(
                experiments.parse_prompt(args.prompt),
                duration=args.duration,
                sample_rate=args.sample_rate,
                seed=args.seed,
            )
            results = experiments.generate_run([settings])
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            files = experiments.write_run(args.output, results)
            result = {"files": files, "manifest": results[0]["manifest"]}
        elif args.command == "train-demo":
            clean = np.stack([synthesis.sine(i % 5 + 1, 0.25, 64)[:, 0] for i in range(32)])
            noisy, epsilon, times = diffusion.training_pairs(
                clean, diffusion.linear_schedule(20), args.seed
            )
            initial = diffusion.initialize_model(16, args.seed)
            model = diffusion.train_denoiser(initial, noisy, times, epsilon, args.steps)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                handle.write(diffusion.checkpoint_encode(model))
            result = {
                "synthetic": True,
                "baseline": "affine epsilon predictor",
                "initial_loss": diffusion.training_loss(initial, noisy, times, epsilon),
                "final_loss": diffusion.training_loss(model, noisy, times, epsilon),
                "checkpoint": str(output),
            }
        else:
            samples, rate = audio.wav_decode(Path(args.path).read_bytes())
            result = {
                "frames": len(samples),
                "channels": samples.shape[1],
                "sample_rate": rate,
                "duration": audio.duration(samples, rate),
                "peak": audio.peak(samples),
                "rms": audio.rms(samples),
            }
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
