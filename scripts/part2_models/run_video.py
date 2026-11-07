"""Run vda_relative by default; model loading, timing and saving are shared."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _video_inference import main

if __name__ == "__main__":
    main(models=('vda_relative', 'vda_metric'), default='vda_relative')
