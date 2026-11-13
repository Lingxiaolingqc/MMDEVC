"""Verify a development checkpoint through RGB-only official inference."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from depth_project.atomic_io import replace_file
from depth_project.lora_inference import LoRADepthPredictor
from depth_project.metric_training import load_development_pack, write_json
from depth_project.metrics import average_image_metrics, evaluate_nyu
from depth_project.records import run_provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--shared-gpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    if args.shared_gpu:
        torch.cuda.set_per_process_memory_fraction(0.125)
    pack = load_development_pack(root)
    folder = root / "runs/metric-research" / args.run
    summary = json.loads((folder / "summary.json").read_text())
    config = summary["config"]
    scale = summary.get("scale", config.get("scale"))
    specification = {"rank": config["rank"], "members": [{"checkpoint": str((folder / "best.safetensors").relative_to(root)),
                     "sha256": summary["checkpoint_sha256"], "scale": scale}]}
    predictor = LoRADepthPredictor(root, specification, pack["identities"]["checkpoint_sha256"])
    ids = config["validation_ids"]
    if not ids or not set(ids).issubset(pack["ids"]):
        raise ValueError("Replay requires the run's nonempty development validation IDs")
    rows, predictions = [], []
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        for index in ids:
            rgb = np.ascontiguousarray(dataset["images"][index].transpose(2, 1, 0))
            prediction = predictor.predict_rgb(rgb).values
            predictions.append(prediction)
            rows.append({"image_id": index, "scene": pack["scenes"][index],
                         **evaluate_nyu(prediction, dataset["depths"][index].T)})
    metrics = average_image_metrics(rows)
    if abs(metrics["abs_rel"] - summary["best_abs_rel"]) > 1e-7:
        raise ValueError(f"RGB-only checkpoint replay differs from selection score: {metrics['abs_rel']}")
    temporary = folder / "validation-inference-replay.pending.npz"
    np.savez_compressed(temporary, image_ids=np.array(ids), depth_m=np.stack(predictions))
    replace_file(temporary, folder / "validation-inference-replay.npz")
    result = {"metrics": metrics, "rows": rows, "specification": specification,
              "validation_ids": ids, "provenance": run_provenance(root),
              "concurrent_gpu_work": args.shared_gpu, "source": "RGB-only official infer with saved LoRA weights"}
    write_json(folder / "validation-inference-replay.json", result)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
