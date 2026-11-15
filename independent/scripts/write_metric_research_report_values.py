"""Write report tables directly from completed frozen evaluation records."""
import json
from pathlib import Path
import shutil

from depth_project.records import sha256_file


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"control": "RGB residual control", "features-gate": "Native features + gate",
          "lora-r8-high": "LoRA, LR $10^{-4}$", "lora-r8-low": "LoRA, LR $3\\times10^{-5}$",
          "lora-r8-high-flip": "LoRA + flip, LR $10^{-4}$",
          "lora-r8-low-flip": "LoRA + flip, LR $3\\times10^{-5}$",
          "lora-r8-high-feature-mean": "LoRA/feature mean, high LR",
          "lora-r8-low-feature-mean": "LoRA/feature mean, low LR"}


def write_table(path, caption, columns, header, rows):
    lines = [r"\begin{table}[t]", r"\centering\small", "\\caption{" + caption + "}",
             "\\begin{tabular}{" + columns + "}", r"\toprule", header + r"\\", r"\midrule"]
    lines.extend(row + r"\\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    run = ROOT / "runs/metric-research-final"
    summary = json.loads((run / "summary.json").read_text())
    benchmark = json.loads((ROOT / "results/metric-research/isolated-benchmark.json").read_text())
    config_path = ROOT / "configs/metric-research-final.json"
    if summary["image_count"] != 654 or summary["configuration_sha256"] != sha256_file(config_path):
        raise ValueError("Report requires the complete identified frozen evaluation")
    if benchmark["configuration_sha256"] != summary["configuration_sha256"]:
        raise ValueError("Runtime and accuracy configuration identities differ")
    config = summary["config"]
    metrics = summary["metrics"]
    base = summary["comparisons"]["baseline_to_selected"]
    previous = summary["comparisons"]["previous_refiner_to_selected"]
    refit_gain = summary["comparisons"]["pre_refit_to_selected"]["relative_abs_rel_reduction_percent"]
    direction = "decreases" if refit_gain >= 0 else "increases"
    values = {"ResearchMethod": config["method"],
              "ResearchBaseAbs": f"{metrics['baseline']['abs_rel']:.5f}",
              "ResearchFinalAbs": f"{metrics['selected']['abs_rel']:.5f}",
              "ResearchGain": f"{base['relative_abs_rel_reduction_percent']:.2f}",
              "ResearchPreviousGain": f"{previous['relative_abs_rel_reduction_percent']:.2f}",
              "ResearchGainCI": "[" + ", ".join(f"{value:.2f}" for value in base["scene_bootstrap_95_interval_percent"]) + r"]\%",
              "ResearchImproved": str(base["improved_images"]), "ResearchWorsened": str(base["worsened_images"]),
              "ResearchRefitComparison": f"AbsRel {direction} by {abs(refit_gain):.2f}\\% after the all-development refit",
              "ResearchSeeds": ", ".join(str(seed) for seed in config["chosen_seeds"]),
              "ResearchBaseMS": f"{1000 * benchmark['methods']['baseline_default']['median_image_s']:.1f}",
              "ResearchFinalMS": f"{1000 * benchmark['methods']['selected']['median_image_s']:.1f}"}
    report = ROOT / "report"
    (report / "metric-research-values.tex").write_text(
        "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in values.items()) + "\n", encoding="utf-8")
    test_labels = {"baseline": "Raw UniDepthV2-S", "previous_refiner": "Previous residual refiner",
                   "pre_refit": "Selected (577 fit)", "selected": "Final (795 fit)"}
    rows = [f"{label} & {metrics[key]['abs_rel']:.5f} & {metrics[key]['rmse']:.5f} & {metrics[key]['delta1']:.5f}"
            for key, label in test_labels.items()]
    write_table(report / "metric-research-test-table.tex", "Complete 654-image NYU test comparison. Lower AbsRel/RMSE and higher $\\delta_1$ are better.",
                "lrrr", r"Method & AbsRel & RMSE (m) & $\delta_1$", rows)
    folds = json.loads((ROOT / "results/metric-research/scene-fold-selection.json").read_text())
    feature_epochs = next(item["fixed_refit_epochs"] for item in folds["ranking"] if item["candidate"] == "features-gate")
    rows = []
    for item in folds["ranking"]:
        epochs = str(item["fixed_refit_epochs"])
        if item["candidate"].endswith("-feature-mean"):
            epochs += f"/{feature_epochs}"
        rows.append(f"{LABELS[item['candidate']]} & {item['mean_fold_metrics']['abs_rel']:.5f} & {epochs}")
    write_table(report / "metric-research-fold-table.tex", "Equal means over three fitting-scene folds. Refit durations use median best fold epochs. Combined rows list LoRA/refiner durations.",
                "lrr", "Candidate & Mean AbsRel & Epochs", rows)
    seeds = json.loads((ROOT / "results/metric-research/seed-selection.json").read_text())
    candidates = [seeds["control"], *seeds["ranking"]]
    rows = [f"{LABELS[item['candidate']]} & {item['mean_seed_metrics']['abs_rel']:.5f} & {item['seed_abs_rel_standard_deviation']:.5f}"
            for item in candidates]
    write_table(report / "metric-research-seed-table.tex", "Three-seed selection results on the same 218 development images. SD describes training-seed variability.",
                "lrr", "Candidate & Mean AbsRel & SD", rows)
    (report / "metric-research-selection-tables.tex").write_text(
        "\\input{metric-research-fold-table}\n\\input{metric-research-seed-table}\n", encoding="utf-8")
    runtime_labels = {"baseline_default": "Raw, default", "baseline_4": "Raw, resolution 4", "baseline_9": "Raw, resolution 9",
                      "previous_refiner": "Previous refiner", "pre_refit": "Selected (577 fit)", "selected": "Final (795 fit)"}
    rows = []
    for name, label in runtime_labels.items():
        record = benchmark["methods"][name]
        rows.append(f"{label} & {1000 * record['median_image_s']:.1f} & {record['peak_allocated_mib']:.0f} & {record['peak_reserved_mib']:.0f}")
    write_table(report / "metric-research-runtime-table.tex", "Isolated RGB-to-depth inference. Memory columns are peak allocated (A) and reserved (R) MiB.",
                "lrrr", "Method & ms/img & A & R", rows)
    destination = ROOT / "results/metric-research"
    shutil.copyfile(run / "summary.json", destination / "final-test.json")
    for name in test_labels:
        shutil.copyfile(run / f"{name}-per-image.csv", destination / f"{name}-test-per-image.csv")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
