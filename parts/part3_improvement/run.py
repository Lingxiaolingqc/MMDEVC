"""part3_improvement: course-facing command entry; implementation remains in scripts/src."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _entry import dispatch

COMMANDS = {
    "flow": [
        "prepare_flow.py",
        []
    ],
    "fusion": [
        "run_fusion_suite.py",
        []
    ],
    "verify": [
        "verify_fusion_suite.py",
        []
    ],
    "freeze": [
        "freeze_fusion_config.py",
        []
    ],
    "compare": [
        "render_depth_video.py",
        []
    ]
}

if __name__ == "__main__":
    raise SystemExit(dispatch("part3_improvement", COMMANDS))
