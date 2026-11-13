"""Recompute final accuracy from identified float predictions and NYU labels."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from .atomic_io import replace_file
from .data import load_nyu_split
from .metric_experiments import verify_frozen_dataset
from .metric_selection import paired_scene_comparison
from .metrics import average_image_metrics, evaluate_nyu
from .records import sha256_file


VARIANTS = ("baseline", "previous_refiner", "pre_refit", "selected")


def evaluate_metric_packets(root, config_path, output):
    root, output = Path(root), Path(output)
    config = json.loads(Path(config_path).read_text())
    verify_frozen_dataset(root, config)
    official_ids = load_nyu_split(root / "data/nyu/splits.mat")["test"].tolist()
    if config["test_ids"] != official_ids:
        raise ValueError("Frozen test image identities differ from the official split")
    configuration_hash = sha256_file(config_path)
    rows = {name: [] for name in VARIANTS}
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        for index in official_ids:
            truth = dataset["depths"][index].T
            scene = "".join(chr(int(value)) for value in dataset[dataset["scenes"][0, index]][:].reshape(-1))
            with np.load(output / f"prediction_{index:04d}.npz", allow_pickle=False) as packet:
                if int(packet["image_id"]) != index:
                    raise ValueError("Saved prediction image identity differs")
                if str(packet["configuration_sha256"]) != configuration_hash:
                    raise ValueError("Saved prediction configuration differs")
                for name in VARIANTS:
                    prediction = packet[f"{name}_m"]
                    if prediction.shape != (480, 640):
                        raise ValueError("Saved prediction must have the course output dimensions")
                    rows[name].append({"image_id": index, "scene": scene, **evaluate_nyu(prediction, truth)})
    for name, values in rows.items():
        destination = output / f"{name}-per-image.csv"
        temporary = destination.with_suffix(".pending.csv")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        replace_file(temporary, destination)
    comparisons = {f"{name}_to_selected": paired_scene_comparison(rows[name], rows["selected"])
                   for name in VARIANTS[:-1]}
    return {"metrics": {name: average_image_metrics(values) for name, values in rows.items()},
            "comparisons": comparisons, "image_count": len(official_ids), "test_ids": official_ids,
            "configuration_sha256": configuration_hash,
            "protocol": "Course crop and strict GT support, raw metric depth, equal-image mean, no test alignment"}
