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
    stereo=commands.add_parser('stereo')
    stereo.add_argument('--data-root',type=Path,default=Path('data/middlebury/MiddEval3/trainingQ'))
    stereo.add_argument('--output',type=Path,default=Path('runs/stereo'))
    baseline=commands.add_parser('baseline')
    baseline.add_argument('--input',type=Path,required=True)
    baseline.add_argument('--output',type=Path,required=True)
    baseline.add_argument('--model',choices=['dav2','vda'],required=True)
    baseline.add_argument('--input-size',type=int,default=518)
    baseline.add_argument('--max-res',type=int,default=640)
    baseline.add_argument('--limit',type=int,default=100)
    baseline.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    nyu=commands.add_parser('nyu')
    nyu.add_argument('--output',type=Path,default=Path('runs/nyu'))
    sweep=commands.add_parser('sweep')
    sweep.add_argument('--input',type=Path,required=True)
    sweep.add_argument('--baseline',type=Path,required=True)
    sweep.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    root=Path(__file__).resolve().parents[2]
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
    if args.command=='stereo':
        from .experiments import run_stereo
        run_stereo(args.data_root,args.output)
    elif args.command=='baseline':
        from .experiments import run_video_model
        run_video_model(args.input,args.output,args.model,root,limit=args.limit,max_res=args.max_res,input_size=args.input_size,precision=args.precision)
    elif args.command=='nyu':
        from .experiments import run_nyu
        run_nyu(root,args.output)
    elif args.command=='sweep':
        from .evaluation import run_development_sweep
        result=run_development_sweep(args.input,args.baseline,args.output)
        print(json.dumps(result['selected'],indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
