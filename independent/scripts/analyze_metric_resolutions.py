"""Evaluate official resolution choices using development images only."""
import json
from pathlib import Path

import numpy as np

from depth_project.metric_research import fit_subset_scale
from depth_project.metric_training import load_development_pack, write_json
from depth_project.metrics import evaluate_nyu


def main():
    root = Path(__file__).resolve().parents[1]
    pack = load_development_pack(root)
    positions = {index: position for position, index in enumerate(pack["ids"])}
    truths = {index: pack["truth"][positions[index]] for index in pack["fit_ids"]}
    records = []
    for resolution in ["default", "4", "9"]:
        folder = root / "runs/metric-research" / f"foundation-{resolution}"
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["identities"]["resolution"] != resolution:
            raise ValueError("Cached resolution differs from its named variant")
        if manifest["completed"] != len(pack["ids"]) or manifest["identities"]["ids"] != pack["ids"]:
            raise ValueError("Resolution cache must cover the complete official development split")
        for key in ["dataset_sha256", "split_sha256", "checkpoint_sha256"]:
            if manifest["identities"][key] != pack["identities"][key]:
                raise ValueError("Resolution cache source identity differs")
        maps = {}
        for index in pack["ids"]:
            with np.load(folder / f"prediction_{index:04d}.npz", allow_pickle=False) as packet:
                values = packet["depth_m"]
                if int(packet["image_id"]) != index or values.shape != (480, 640) or not np.isfinite(values).all():
                    raise ValueError("Resolution prediction identity or values differ")
                maps[index] = values
        scale = fit_subset_scale(maps, truths, pack["fit_ids"])
        for label, multiplier in [("raw", 1.0), ("fit-only-global-calibration", scale)]:
            rows = [{"image_id": index, "scene": pack["scenes"][index],
                     **evaluate_nyu(maps[index] * multiplier, pack["truth"][positions[index]])}
                    for index in pack["validation_ids"]]
            record = {"resolution": resolution, "processed_hw": manifest["processed_hw"],
                      "variant": label, "scale": multiplier, "abs_rel": float(np.mean([row["abs_rel"] for row in rows])),
                      "rmse": float(np.mean([row["rmse"] for row in rows])),
                      "delta1": float(np.mean([row["delta1"] for row in rows])), "rows": rows}
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "rows"}), flush=True)
    result = {"fit_ids": pack["fit_ids"], "validation_ids": pack["validation_ids"], "records": records,
              "scope": "Development comparison of unchanged RGB-only official resolution variants"}
    write_json(root / "results/metric-research/resolutions.json", result)


if __name__ == "__main__":
    main()
