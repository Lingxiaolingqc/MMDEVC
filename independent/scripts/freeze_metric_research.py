"""Freeze the selected recipe after its fixed all-development refits."""
import json
from pathlib import Path
import shutil

from depth_project.atomic_io import replace_file
from depth_project.data import load_nyu_split
from depth_project.metric_training import write_json
from depth_project.records import run_provenance, sha256_file, sha256_source


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "configs/metric-research-final.json"
    if output.exists():
        raise FileExistsError("Research configuration is already frozen")
    selection = json.loads((root / "results/metric-research/seed-selection.json").read_text())
    decision = json.loads((root / "results/metric-research/refit-decision.json").read_text())
    manifest = json.loads((root / "runs/nyu-development/manifest.json").read_text())
    split = load_nyu_split(root / "data/nyu/splits.mat")
    winner = selection["winner"]
    if not winner.startswith("lora-"):
        raise ValueError("Selected family requires a corresponding RGB-only inference implementation")
    if decision["winner"] != winner or decision["chosen_seeds"] != selection["chosen_seeds"]:
        raise ValueError("Refit decision differs from completed seed selection")
    recipe = winner.removesuffix("-flip").removesuffix("-feature-mean")
    combined = winner.endswith("-feature-mean")
    families = [("lora", recipe)] + ([("refiner", "features-gate")] if combined else [])
    refit_jobs = [job for family, _ in families for job in decision["jobs"][family]]
    if any(len(decision["jobs"][family]) != len(selection["chosen_seeds"]) for family, _ in families):
        raise ValueError("Final refit jobs differ from the selected inference family")
    specifications = {}
    artifacts = root / "artifacts/metric-research"
    artifacts.mkdir(parents=True, exist_ok=True)
    for scope, prefix in [("pre_refit", "selection"), ("selected", "refit")]:
        components = {}
        for family, component_recipe in families:
            members = []
            ranks = set()
            model_options = None
            for seed in selection["chosen_seeds"]:
                name = f"{prefix}-{component_recipe}-seed{seed}"
                folder = root / "runs/metric-research" / name
                summary = json.loads((folder / "summary.json").read_text())
                config = summary["config"]
                expected_fit = manifest["fit_ids"] if scope == "pre_refit" else sorted(split["development"].tolist())
                expected_validation = manifest["validation_ids"] if scope == "pre_refit" else []
                if config["seed"] != seed or config["fit_ids"] != expected_fit or config["validation_ids"] != expected_validation:
                    raise ValueError(f"Selected checkpoint partition or seed differs for {name}")
                if family == "lora" and (not summary["frozen_parameters_unchanged"] or summary["initial_maximum_baseline_difference_m"] != 0):
                    raise ValueError(f"Foundation preservation checks failed for {name}")
                if scope == "selected":
                    job = next(item for item in decision["jobs"][family] if item["seed"] == seed)
                    if config != job or summary["best_epoch"] != job["max_epochs"]:
                        raise ValueError(f"Fixed all-development refit is incomplete for {name}")
                checkpoint = folder / "best.safetensors"
                digest = sha256_file(checkpoint)
                if digest != summary["checkpoint_sha256"]:
                    raise ValueError(f"Selected checkpoint identity differs for {name}")
                artifact = artifacts / f"{scope}-{family}-seed{seed}.safetensors"
                if artifact.exists() and sha256_file(artifact) != digest:
                    raise ValueError("Existing distributed weights differ from the selected checkpoint")
                if not artifact.exists():
                    temporary = artifact.with_suffix(".pending.safetensors")
                    shutil.copyfile(checkpoint, temporary)
                    replace_file(temporary, artifact)
                members.append({"seed": seed, "checkpoint": artifact.relative_to(root).as_posix(),
                                "sha256": digest, "scale": summary["scale"]})
                if family == "lora":
                    ranks.add(config["rank"])
                elif model_options is not None and model_options != config["model"]:
                    raise ValueError("Selected feature checkpoints have different architectures")
                else:
                    model_options = config["model"]
            if family == "lora":
                if len(ranks) != 1:
                    raise ValueError("Selected seed checkpoints have different LoRA ranks")
                components[family] = {"rank": ranks.pop(), "flip_weight": 0.25 if winner.endswith("-flip") else 0, "members": members}
            else:
                components[family] = {"model": model_options, "feature_hw": [240, 320], "flip_weight": 0.25, "members": members}
        specifications[scope] = {"type": "lora-feature-mean", **components} if combined else components["lora"]
    identities = manifest["identities"]
    source_names = ["cli", "research_evaluation", "research_inference", "lora_inference", "lora_training", "metric_adaptation", "models",
                    "metric_experiments", "metric_packet_evaluation", "metric_reference", "metric_research",
                    "metric_selection", "metric_training", "metrics", "refinement", "residual_refiner", "data", "records", "atomic_io"]
    source_hashes = {f"src/depth_project/{name}.py": sha256_source(root / f"src/depth_project/{name}.py") for name in source_names}
    rank = (specifications["selected"]["lora"] if combined else specifications["selected"])["rank"]
    method = f"rank-{rank} Q/V LoRA"
    if combined:
        method += " and gated feature-refiner averaging"
    if winner.endswith("-flip"):
        method += " with fixed flip averaging"
    config = {"format_version": 2, "method": method,
              "selection_status": "frozen after development selection and full-development refit",
              "foundation_checkpoint_sha256": identities["checkpoint_sha256"],
              "dataset_sha256": identities["dataset_sha256"], "split_sha256": identities["split_sha256"],
              "selection_fit_ids": manifest["fit_ids"], "selection_validation_ids": manifest["validation_ids"],
              "test_ids": split["test"].tolist(), "chosen_seeds": selection["chosen_seeds"], "winner": winner,
              "refit_jobs": refit_jobs, **specifications,
              "previous_config_sha256": sha256_file(root / "configs/metric-final.json"),
              "selection_evidence_sha256": {name: sha256_file(root / "results/metric-research" / name)
                                             for name in ["scene-fold-selection.json", "seed-selection.json", "refit-decision.json"]},
              "inference_source_sha256": source_hashes, "source_hash_newlines": "LF", "provenance": run_provenance(root),
              "supervision": "NYU development adaptation, RGB-only test inference, raw metric output, no test scale fitting",
              "test_history": "Original reference and previous refiner test results were inspected before this study",
              "refit_test_order_rule": decision["test_order_rule"]}
    from depth_project.research_evaluation import verify_recipe
    verify_recipe(root, config)
    write_json(output, config)
    print(json.dumps({"method": method, "chosen_seeds": selection["chosen_seeds"],
                      "fixed_refit_epochs": [job["max_epochs"] for job in refit_jobs],
                      "configuration_sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
