"""application: course-facing command entry; implementation remains in scripts/src."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _entry import dispatch

COMMANDS = {
    "blur": [
        "render_blur_demo.py",
        []
    ],
    "package": [
        "package_depth_results.py",
        []
    ]
}

if __name__ == "__main__":
    raise SystemExit(dispatch("application", COMMANDS))
