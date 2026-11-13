"""Atomic artifact replacement tolerant of short-lived Windows file handles."""
from pathlib import Path
import time


def replace_file(source, destination):
    source = Path(source)
    for attempt in range(8):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.1 * 2 ** attempt)
