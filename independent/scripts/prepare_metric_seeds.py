"""Prepare repeated-seed jobs for the two scene-CV finalists and control."""
import json
from pathlib import Path

from depth_project.metric_training import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    selection = json.loads((root / "results/metric-research/scene-fold-selection.json").read_text())
    manifest = json.loads((root / "runs/nyu-development/manifest.json").read_text())
    recipes = {"control"}
    for candidate in selection["advanced_candidates"]:
        recipes.add(candidate.removesuffix("-flip").removesuffix("-feature-mean"))
        if candidate.endswith("-feature-mean"):
            recipes.add("features-gate")
    jobs = {"refiner": [], "lora": []}
    for recipe in sorted(recipes):
        source = root / "runs/metric-research" / f"{recipe}-fold0/summary.json"
        template = json.loads(source.read_text())["config"]
        family = "lora" if recipe.startswith("lora-") else "refiner"
        for seed in [320112, 320113, 320114]:
            job = {**template, "name": f"selection-{recipe}-seed{seed}", "seed": seed,
                   "fit_ids": manifest["fit_ids"], "validation_ids": manifest["validation_ids"]}
            jobs[family].append(job)
    for family, items in jobs.items():
        if items:
            output = root / ".cache/metric-research" / f"seeds-{family}-jobs.json"
            if output.exists() and json.loads(output.read_text()) != items:
                raise ValueError("Existing repeated-seed jobs have different settings")
            write_json(output, items)
    print(json.dumps({"advanced_candidates": selection["advanced_candidates"],
                      "training_recipes": sorted(recipes), "job_counts": {key: len(value) for key, value in jobs.items()}}, indent=2))


if __name__ == "__main__":
    main()
