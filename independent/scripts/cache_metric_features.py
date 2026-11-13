"""Cache official development-only features and resolution variants."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from depth_project.data import load_nyu_split
from depth_project.metric_adaptation import infer_differentiable
from depth_project.metric_training import write_json
from depth_project.models import load_image_adapter
from depth_project.records import run_provenance, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", default="default", choices=["default", "4", "9"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--projection", action="store_true")
    parser.add_argument("--shared-gpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.shared_gpu:
        torch.cuda.set_per_process_memory_fraction(0.25)
    torch.set_num_threads(4)
    adapter = load_image_adapter("unidepth", root)
    model = adapter.model
    if args.resolution != "default":
        model.resolution_level = int(args.resolution)
    ids = sorted(load_nyu_split(root / "data/nyu/splits.mat")["development"].tolist())
    output = root / f"runs/metric-research/foundation-{args.resolution}"
    output.mkdir(parents=True, exist_ok=True)
    development = json.loads((root / "runs/nyu-development/manifest.json").read_text())
    data_identities = development["identities"]
    for key, path in [("dataset_sha256", "data/nyu/nyu_depth_v2_labeled.mat"), ("split_sha256", "data/nyu/splits.mat")]:
        if sha256_file(root / path) != data_identities[key]:
            raise ValueError("Feature-cache dataset identity differs")
    identities = {**adapter.metadata, "resolution": args.resolution, "ids": ids, "projection": args.projection,
                  "dataset_sha256": data_identities["dataset_sha256"], "split_sha256": data_identities["split_sha256"]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identities"] != identities:
        raise ValueError("Feature cache identity differs")
    captured = {}

    def record_input(module, inputs):
        captured["processed_hw"] = list(inputs[0].shape[-2:])

    def record_projection(module, inputs):
        captured["projection"] = inputs[0].detach()

    def record_rays(module, inputs, outputs):
        captured["rays"] = outputs["rays"].detach()

    handles = [model.pixel_encoder.register_forward_pre_hook(record_input)]
    if args.projection:
        handles.append(model.pixel_decoder.depth_layer.to_depth_lr.register_forward_pre_hook(record_projection))
        handles.append(model.pixel_decoder.register_forward_hook(record_rays))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "identities": identities, "provenance": run_provenance(root), "concurrent_gpu_work": args.shared_gpu,
        "maximum_default_depth_difference_m": 0.0, "maximum_differentiable_difference_m": None}
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        for number, index in enumerate(ids[:args.limit]):
            path = output / f"prediction_{index:04d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if int(saved["image_id"]) != index or not np.isfinite(saved["depth_features"]).all():
                        raise ValueError("Feature cache image identity or values differ")
                continue
            rgb = np.ascontiguousarray(dataset["images"][index].transpose(2, 1, 0))
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).cuda()
            with torch.inference_mode():
                original = model.infer(tensor)
                values = {"depth_m": original["depth"][0, 0].float().cpu().numpy(),
                          "confidence": original["confidence"][0, 0].float().cpu().numpy(),
                          "depth_features": original["depth_features"][0].half().cpu().numpy(),
                          "processed_hw": np.array(captured["processed_hw"]), "image_id": index}
                if args.projection:
                    values["projection_features"] = captured["projection"][0].half().cpu().numpy()
                    height, width = captured["processed_hw"]
                    values["ray_z"] = captured["rays"][0, :, 2].reshape(height, width).float().cpu().numpy()
                if number == 0:
                    with torch.autocast("cuda", dtype=torch.float16):
                        reproduced = infer_differentiable(model, tensor)
                    difference = (reproduced - original["depth"]).abs().max().item()
                    manifest["maximum_differentiable_difference_m"] = difference
                    if difference != 0:
                        raise ValueError(f"Differentiable inference differs before adaptation by {difference}")
                flipped = model.infer(torch.flip(tensor, [-1]))
                values["flip_m"] = torch.flip(flipped["depth"], [-1])[0, 0].float().cpu().numpy()
            if args.resolution == "default":
                with np.load(root / f"runs/nyu-development/prediction_{index:04d}.npz") as old:
                    difference = float(np.max(np.abs(values["depth_m"] - old["depth_m"])))
                manifest["maximum_default_depth_difference_m"] = max(manifest["maximum_default_depth_difference_m"], difference)
                if difference != 0:
                    raise ValueError("Official default predictions differ from the recorded baseline")
            temporary = path.with_suffix(".pending.npz")
            np.savez_compressed(temporary, **values)
            temporary.replace(path)
            manifest["feature_shape"] = list(values["depth_features"].shape)
            manifest["processed_hw"] = values["processed_hw"].tolist()
            if args.projection:
                manifest["projection_shape"] = list(values["projection_features"].shape)
            manifest["completed"] = len(list(output.glob("prediction_*.npz")))
            write_json(manifest_path, manifest)
            if number == 0 or (number + 1) % 50 == 0:
                print(json.dumps({key: manifest[key] for key in ["completed", "feature_shape", "processed_hw"]}), flush=True)
    for handle in handles:
        handle.remove()
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
