"""part1_baseline: course-facing command entry; implementation remains in scripts/src."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _entry import dispatch

COMMANDS = {
    "stereo": [
        "run_stereo_suite.py",
        []
    ],
    "stereo-sample": [
        "run_stereo_baseline.py",
        []
    ],
    "audit-stereo": [
        "audit_course_stereo.py",
        []
    ],
    "framewise": [
        "run_video_models.py",
        [
            "--model",
            "dav2_relative"
        ]
    ]
}

if __name__ == "__main__":
    raise SystemExit(dispatch("part1_baseline", COMMANDS))
