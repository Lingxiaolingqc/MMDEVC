"""Run the declared scene-separated learning curves and refiner candidates."""
import argparse
import json
from pathlib import Path

import torch

from depth_project.metric_research import nested_scene_subsets, scene_folds
from depth_project.metric_training import load_development_pack, train_candidate, write_json


def candidate_jobs(pack, stage):
    fitting = pack["fit_ids"]
    validation = pack["validation_ids"]
    scenes = {index: pack["scenes"][index] for index in fitting}
    seeds = [320112, 320113, 320114]
    jobs = []
    if stage == "curves":
        for seed in seeds:
            for target, ids in nested_scene_subsets(scenes, [100, 200, 400, len(fitting)], seed).items():
                jobs.append({"name": f"curve-{target}-seed{seed}", "fit_ids": ids, "validation_ids": validation,
                             "seed": seed, "max_steps": 6000, "max_epochs": 1000,
                             "evaluate_every": 250, "minimum_steps": 1000, "patience": 8,
                             "batch_size": 8, "measure_fitting": True, "target_image_count": target})
    else:
        settings = {
            "control": {},
            "rgb-augmentation": {"augment_rgb": True},
            "rgb-dropout": {"augment_rgb": True, "model": {"dropout": 0.1}},
            "narrow": {"augment_rgb": True, "model": {"widths": [8, 16, 32, 48], "dropout": 0.1}},
            "full-resolution": {"full_resolution": True},
            "full-gradient": {"full_resolution": True, "gradient_weight": 0.05},
        }
        if stage == "controls":
            for name, options in settings.items():
                jobs.append({"name": name, "fit_ids": fitting, "validation_ids": validation,
                             "seed": seeds[0], "max_epochs": 80, "patience": 10,
                             "batch_size": 8, **options})
        elif stage == "folds":
            folds = scene_folds(scenes, 3, 320113)
            for name, options in settings.items():
                for number, held_out in enumerate(folds):
                    ids = sorted(set(fitting) - set(held_out))
                    jobs.append({"name": f"{name}-fold{number}", "fit_ids": ids, "validation_ids": held_out,
                                 "seed": seeds[0], "max_epochs": 80, "patience": 10,
                                 "batch_size": 8, **options})
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["curves", "controls", "folds", "config"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--only", help="Run names containing this exact text")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pack = load_development_pack(root)
    jobs = json.loads(args.config.read_text()) if args.config else candidate_jobs(pack, args.stage)
    if args.only:
        jobs = [job for job in jobs if args.only in job["name"]]
    if not jobs:
        raise ValueError("No candidate jobs matched")
    output = root / "runs/metric-research"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{args.stage}-jobs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != jobs:
        raise ValueError("Stage job manifest already exists with different settings")
    write_json(manifest, jobs)
    summaries = []
    for job in jobs:
        print(f"Starting {job['name']} with {len(job['fit_ids'])} fitting images", flush=True)
        result = train_candidate(pack, job, output / job["name"])
        summaries.append({"name": job["name"], "best_abs_rel": result["best_abs_rel"],
                          "best_epoch": result["best_epoch"], "completed_steps": result["completed_steps"],
                          "fit_image_count": len(job["fit_ids"]), "fit_scene_count": len({pack['scenes'][index] for index in job['fit_ids']}),
                          "seed": job["seed"], "scale": result["scale"]})
        write_json(output / f"{args.stage}-summary.json", summaries)
        torch.cuda.empty_cache()
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
