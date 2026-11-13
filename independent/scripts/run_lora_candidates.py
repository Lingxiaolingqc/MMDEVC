"""Execute explicit scene-fold, seed or fixed-development LoRA jobs."""
import argparse
import json
from pathlib import Path

import torch

from depth_project.lora_training import train_lora_candidate
from depth_project.metric_training import load_development_pack, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shared-gpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.shared_gpu:
        torch.cuda.set_per_process_memory_fraction(0.7)
    pack = load_development_pack(root)
    jobs = json.loads(args.config.read_text())
    if not jobs:
        raise ValueError("No LoRA jobs supplied")
    results = []
    for job in jobs:
        result = train_lora_candidate(root, pack, job, root / "runs/metric-research" / job["name"])
        results.append({"name": job["name"], "best_abs_rel": result["best_abs_rel"],
                        "best_epoch": result["best_epoch"], "completed_epochs": result["completed_epochs"]})
        write_json(root / "runs/metric-research" / f"{args.config.stem}-summary.json", results)
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
