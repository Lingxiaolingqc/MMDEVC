"""Load validated, image-aligned frozen features for development experiments."""
import json
from pathlib import Path

import numpy as np

from .atomic_io import replace_file
from .metric_training import write_json
from .records import sha256_file


def attach_depth_features(root, pack):
    root = Path(root)
    source = root / "runs/metric-research/foundation-default"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("completed") != len(pack["ids"]) or manifest["identities"]["ids"] != pack["ids"]:
        raise ValueError("Complete the frozen feature cache before refiner training")
    for name in ["dataset_sha256", "split_sha256", "checkpoint_sha256"]:
        if manifest["identities"][name] != pack["identities"][name]:
            raise ValueError("Frozen feature source identities differ")
    if manifest["processed_hw"] != [490, 644]:
        raise ValueError("This feature cache expects the recorded default NYU preprocessing")
    folder = root / ".cache/metric-research"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "depth-features.npy"
    complete = path.with_suffix(".json")
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["identities"] != manifest["identities"] or sha256_file(path) != previous["sha256"]:
            raise ValueError("Prepared frozen feature identities differ")
    else:
        temporary = path.with_suffix(".pending.npy")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                          shape=(len(pack["ids"]), *manifest["feature_shape"]))
        for position, index in enumerate(pack["ids"]):
            with np.load(source / f"prediction_{index:04d}.npz", allow_pickle=False) as packet:
                feature = packet["depth_features"]
                if int(packet["image_id"]) != index or not np.isfinite(feature).all():
                    raise ValueError("Frozen feature image or finite-value check failed")
                values[position] = feature
        values.flush()
        del values
        replace_file(temporary, path)
        write_json(complete, {"identities": manifest["identities"], "sha256": sha256_file(path)})
    return {**pack, "depth_features": np.load(path, mmap_mode="c"),
            "identities": {**pack["identities"], "depth_features_sha256": sha256_file(path)}}
