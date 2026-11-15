"""Write report tables directly from completed frozen evaluation records."""
import json
from pathlib import Path
import shutil

from depth_project.records import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def write_table(path, caption, columns, header, rows, wide=False):
    environment = "table*" if wide else "table"
    lines = ["\\begin{" + environment + "}[t]", r"\centering\small", "\\caption{" + caption + "}",
             "\\begin{tabular}{" + columns + "}", r"\toprule", header + r"\\", r"\midrule"]
    lines.extend(row + r"\\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", "\\end{" + environment + "}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_tables(report, seeds, search):
    original_rows = [row for row in search["rows"] if row["config"]["flip_weight"] == 0 and row["config"]["radius"] == 0]
    raw = next(row["abs_rel"] for row in original_rows if row["config"]["scale"] == 1)
    calibrated = next(row["abs_rel"] for row in original_rows if row["config"]["scale"] != 1)
    by_name = {row["candidate"]: row for row in seeds["ranking"]}
    by_name["control"] = seeds["control"]
    families = [("RGB residual refiner", ["control"]), ("Native-feature refiner", ["features-gate"]),
                ("LoRA + calibration", ["lora-r8-high", "lora-r8-low"]),
                ("LoRA + feature-refiner average", ["lora-r8-high-feature-mean", "lora-r8-low-feature-mean"]),
                ("LoRA + calibration + flip", ["lora-r8-high-flip", "lora-r8-low-flip"])]
    rows = [f"Unchanged UniDepthV2-S & {raw:.5f} & ---",
            f"Global scale calibration & {calibrated:.5f} & ---"]
    for label, names in families:
        chosen = min((by_name[name] for name in names), key=lambda item: item["mean_seed_metrics"]["abs_rel"])
        single = f"{chosen['mean_seed_metrics']['abs_rel']:.5f} $\\pm$ {chosen['seed_abs_rel_standard_deviation']:.5f}"
        ensemble = f"{chosen['ensemble_selection']['ensemble_abs_rel']:.5f}"
        if chosen["candidate"] == seeds["winner"]:
            label = "\\textbf{" + label + "}"
            ensemble = "\\textbf{" + ensemble + "}"
        rows.append(f"{label} & {single} & {ensemble}")
    write_table(report / "metric-research-method-table.tex",
                "Candidate pipelines on the same 218 development images, with one selected learning rate per family. Per-seed results give mean and SD across three training seeds. A LoRA/feature pipeline pairs both models within each seed. The prediction average combines three pipeline outputs before scoring. Every learned candidate includes a fitting-only global scale. Residual refiners also use original/flip depth features.",
                "lrr", "Pipeline & Per-seed pipeline AbsRel & Three-seed average AbsRel", rows, wide=True)
    raw_name = seeds["winner"].removesuffix("-flip")
    seed = by_name[raw_name]["ensemble_selection"]["median_seed"]
    single = by_name[raw_name]["ensemble_selection"]["seed_abs_rel"][str(seed)]
    flipped = by_name[raw_name + "-flip"]["ensemble_selection"]["seed_abs_rel"][str(seed)]
    final = by_name[seeds["winner"]]["ensemble_selection"]["ensemble_abs_rel"]
    rows = [f"Raw UniDepthV2-S & {raw:.5f}", f"+ global scale calibration & {calibrated:.5f}",
            f"+ LoRA adaptation & {single:.5f}", f"+ fixed flip averaging & {flipped:.5f}",
            f"+ three-seed averaging & {final:.5f}"]
    write_table(report / "metric-research-component-table.tex",
                f"Cumulative ablation on 218 development images. The two single-adapter rows use the same preselected seed {seed}. LoRA is evaluated on top of frozen calibration. The last row averages three seed models. The full recipe receives the final test comparison.",
                "lr", "Components included & AbsRel", rows)
    (report / "metric-research-selection-tables.tex").write_text(
        "\\input{metric-research-method-table}\n\\input{metric-research-component-table}\n", encoding="utf-8")


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
                   "pre_refit": "Full recipe (577 fit)", "selected": "Full recipe (795 fit)"}
    rows = [f"{label} & {metrics[key]['abs_rel']:.5f} & {metrics[key]['rmse']:.5f} & {metrics[key]['delta1']:.5f}"
            for key, label in test_labels.items()]
    write_table(report / "metric-research-test-table.tex", "Complete 654-image NYU test comparison. Lower AbsRel/RMSE and higher $\\delta_1$ are better.",
                "lrrr", r"Method & AbsRel & RMSE (m) & $\delta_1$", rows)
    seeds = json.loads((ROOT / "results/metric-research/seed-selection.json").read_text())
    search = json.loads((ROOT / "results/metric-development-search.json").read_text())
    write_method_tables(report, seeds, search)
    runtime_labels = {"baseline_default": "Raw, default", "baseline_4": "Raw, resolution 4", "baseline_9": "Raw, resolution 9",
                      "previous_refiner": "Previous refiner", "pre_refit": "Full recipe (577 fit)", "selected": "Full recipe (795 fit)"}
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
