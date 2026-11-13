"""Benchmark frozen metric pipelines in a dedicated GPU execution window."""
import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from depth_project.benchmark import benchmark_metric_images
from depth_project.lora_inference import LoRADepthPredictor
from depth_project.metric_reference import MetricRefinerReference
from depth_project.metric_training import write_json
from depth_project.models import load_image_adapter
from depth_project.records import run_provenance, sha256_file
from depth_project.research_evaluation import verify_recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-count", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.image_count < 1 or args.repeats < 1:
        parser.error("Image count and repeats must be positive")
    if args.output.exists():
        raise FileExistsError("Metric benchmark output already exists")
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    split = verify_recipe(root, config)
    torch.set_num_threads(4)
    images, ids, scenes = [], [], set()
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        for index in split["development"].tolist():
            scene = "".join(chr(int(value)) for value in dataset[dataset["scenes"][0, index]][:].reshape(-1))
            if scene in scenes:
                continue
            images.append(np.ascontiguousarray(dataset["images"][index].transpose(2, 1, 0)))
            ids.append(index)
            scenes.add(scene)
            if len(images) == args.image_count:
                break

    def load_case(name):
        if name.startswith("baseline"):
            adapter = load_image_adapter("unidepth", root)
            if name != "baseline_default":
                level = int(name.rsplit("_", 1)[-1])
                adapter.model.resolution_level = level
                adapter.metadata["preprocessing_id"] = f"official-infer-rgb-only-resolution-level-{level}"
            return adapter
        if name == "previous_refiner":
            return MetricRefinerReference(root)
        return LoRADepthPredictor(root, config[name], config["foundation_checkpoint_sha256"])

    methods = {}
    for name in ["baseline_default", "baseline_4", "baseline_9", "previous_refiner", "pre_refit", "selected"]:
        adapter = load_case(name)
        model = adapter.adapter.model if name == "previous_refiner" else adapter.model
        processed = set()

        def record_input(module, inputs):
            processed.add(tuple(inputs[0].shape[-2:]))

        handle = model.pixel_encoder.register_forward_pre_hook(record_input)
        result = benchmark_metric_images(images, adapter, args.repeats, torch)
        handle.remove()
        methods[name] = {**result, "metadata": adapter.metadata, "processed_hw": [list(shape) for shape in sorted(processed)]}
        print(json.dumps({"method": name, "median_image_s": result["median_image_s"],
                          "peak_allocated_mib": result["peak_allocated_mib"], "processed_hw": methods[name]["processed_hw"]}), flush=True)
        del adapter, model, handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = {"methods": methods, "image_ids": ids, "scene_count": len(scenes), "input_hw": [480, 640],
              "configuration_sha256": sha256_file(args.config), "provenance": run_provenance(root),
              "hardware": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
              "scope": "Dedicated per-method inference benchmark using preloaded development RGB from distinct scenes"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
