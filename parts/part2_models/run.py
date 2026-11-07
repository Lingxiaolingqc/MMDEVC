"""part2_models: course-facing command entry; implementation remains in scripts/src."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _entry import dispatch

COMMANDS = {
    "metric": [
        "run_nyuv2.py",
        [
            "--model",
            "unidepth"
        ]
    ],
    "evaluate-metric": [
        "evaluate_course_nyu.py",
        []
    ],
    "video": [
        "run_video_models.py",
        [
            "--model",
            "vda_relative"
        ]
    ],
    "prepare-video": [
        "prepare_video.py",
        []
    ],
    "compare": [
        "render_depth_video.py",
        []
    ]
}

if __name__ == "__main__":
    raise SystemExit(dispatch("part2_models", COMMANDS))
