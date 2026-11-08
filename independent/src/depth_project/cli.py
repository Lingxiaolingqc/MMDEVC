"""Reproducible command-line entry points; run with python -m depth_project.cli."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2

from .video import inspect_video, iter_rgb_frames


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent AIAA3201 depth project")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-video")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--required-frames", type=int, default=100)
    inspect.add_argument("--preview-frames", type=int, default=3)
    inspect.add_argument("--output", type=Path, default=Path("runs/video-inspection"))
    args = parser.parse_args(argv)
    if args.command == "inspect-video":
        info = inspect_video(args.input, args.required_frames)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "inspection.json").write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
        for frame in iter_rgb_frames(args.input, args.preview_frames):
            target = args.output / f"frame_{frame.index:04d}.png"
            if not cv2.imwrite(str(target), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write {target}")
        print(json.dumps(asdict(info), indent=2))
        return 0 if info.compliant else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
