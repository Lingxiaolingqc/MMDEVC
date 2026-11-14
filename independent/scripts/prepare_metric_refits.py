"""Fix final all-development jobs from completed scene-CV and seed selection."""
import json
from pathlib import Path

from depth_project.data import load_nyu_split
from depth_project.metric_research import validate_refit
from depth_project.metric_training import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    seed_selection = json.loads((root / "results/metric-research/seed-selection.json").read_text())
    folds = json.loads((root / "results/metric-research/scene-fold-selection.json").read_text())
    manifest = json.loads((root / "runs/nyu-development/manifest.json").read_text())
    development = sorted(manifest["fit_ids"] + manifest["validation_ids"])
    split = load_nyu_split(root / "data/nyu/splits.mat")
    if development != sorted(split["development"].tolist()):
        raise ValueError("Final refit identities differ from the official development split")
    winner = seed_selection["winner"]
    recipes = [winner.removesuffix("-flip").removesuffix("-feature-mean")]
    if winner.endswith("-feature-mean"):
        recipes.append("features-gate")
    jobs = {"refiner": [], "lora": []}
    for recipe in recipes:
        cv = next(item for item in folds["ranking"] if item["candidate"] == recipe)
        duration = cv["fixed_refit_epochs"]
        if duration < 1:
            raise ValueError("Selected trained recipe requires a positive fixed refit duration")
        family = "lora" if recipe.startswith("lora-") else "refiner"
        for seed in seed_selection["chosen_seeds"]:
            source = seed_selection["sources"][f"selection-{recipe}-seed{seed}"]
            if source["config"]["seed"] != seed:
                raise ValueError("Selected source seed differs from its recorded identity")
            job = {**source["config"], "name": f"refit-{recipe}-seed{seed}", "fit_ids": development,
                   "validation_ids": [], "max_epochs": duration}
            validate_refit(development, development, split["test"].tolist(), duration, [])
            jobs[family].append(job)
    for family, items in jobs.items():
        if items:
            output = root / ".cache/metric-research" / f"refit-{family}-jobs.json"
            if output.exists() and json.loads(output.read_text()) != items:
                raise ValueError("Existing all-development jobs have different settings")
            write_json(output, items)
    decision = {"status": "fixed before all-development refitting and test evaluation",
                "winner": winner, "chosen_seeds": seed_selection["chosen_seeds"],
                "duration_rule": "Median best epoch of three fitting-scene folds, rounded half up",
                "jobs": jobs, "test_order_rule": "Retain the all-795 refit as final regardless of its ordering against the pre-refit model on test"}
    write_json(root / "results/metric-research/refit-decision.json", decision)
    print(json.dumps({"winner": winner, "jobs": {key: [(job["name"], job["max_epochs"]) for job in items]
                                               for key, items in jobs.items()}}, indent=2))


if __name__ == "__main__":
    main()
