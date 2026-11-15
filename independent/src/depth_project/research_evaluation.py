"""Frozen RGB-only research inference followed by independent NYU scoring."""
import json
from pathlib import Path
import time

from filelock import FileLock
import h5py
import numpy as np
import torch

from .atomic_io import replace_file
from .data import load_nyu_split
from .research_inference import load_research_predictor
from .metric_experiments import verify_frozen_dataset
from .metric_packet_evaluation import VARIANTS, evaluate_metric_packets
from .metric_reference import MetricRefinerReference
from .metric_research import validate_refit
from .metric_training import write_json
from .records import run_provenance, sha256_file


def verify_recipe(root, config):
    if config["selection_status"] != "frozen after development selection and full-development refit":
        raise ValueError("The complete research recipe must be frozen before test inference")
    verify_frozen_dataset(root, config)
    split = load_nyu_split(root / "data/nyu/splits.mat")
    development, test = split["development"].tolist(), split["test"].tolist()
    selected = config["selection_fit_ids"] + config["selection_validation_ids"]
    if len(selected) != len(development) or set(selected) != set(development):
        raise ValueError("Development selection must cover the official development split exactly once")
    if config["test_ids"] != test:
        raise ValueError("Frozen test identities differ from the official split")
    if not config["refit_jobs"]:
        raise ValueError("The selected adapted model requires recorded full-development refits")
    for job in config["refit_jobs"]:
        validate_refit(job["fit_ids"], development, test, job["max_epochs"], job["validation_ids"])
    if sha256_file(root / "configs/metric-final.json") != config["previous_config_sha256"]:
        raise ValueError("Previous comparison configuration identity differs")
    for relative, expected in config["inference_source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"Frozen inference source identity differs for {relative}")
    return split


def verify_packet(path, image_id, configuration_hash):
    with np.load(path, allow_pickle=False) as packet:
        if int(packet["image_id"]) != image_id or str(packet["configuration_sha256"]) != configuration_hash:
            raise ValueError("Existing prediction image or configuration identity differs")
        for name in VARIANTS:
            values = packet[f"{name}_m"]
            if values.shape != (480, 640) or not np.isfinite(values).all():
                raise ValueError("Existing prediction dimensions or values differ")


def run_research_evaluation(root, config_path, output):
    root, output = Path(root), Path(output)
    config = json.loads(Path(config_path).read_text())
    split = verify_recipe(root, config)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        return _run(root, config_path, config, split, output)


def _run(root, config_path, config, split, output):
    if (output / "summary.json").exists():
        raise FileExistsError("The completed research evaluation already exists")
    configuration_hash = sha256_file(config_path)
    run_path = output / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {
        "configuration_sha256": configuration_hash, "provenance": run_provenance(root), "prediction_sha256": {}}
    if run["configuration_sha256"] != configuration_hash:
        raise ValueError("Incomplete evaluation configuration identity differs")
    write_json(run_path, run)
    torch.set_num_threads(4)
    foundation_hash = config["foundation_checkpoint_sha256"]
    predictors = {name: load_research_predictor(root, config[name], foundation_hash) for name in ["pre_refit", "selected"]}
    reference = MetricRefinerReference(root)
    cuda = predictors["selected"].device == "cuda"
    reused, written = 0, 0
    start = time.perf_counter()
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        warm = np.ascontiguousarray(dataset["images"][int(split["development"][0])].transpose(2, 1, 0))
        for _ in range(2):
            reference.predict_pair(warm)
            for predictor in predictors.values():
                predictor.predict_rgb(warm)
        for number, index in enumerate(config["test_ids"]):
            path = output / f"prediction_{index:04d}.npz"
            key = str(index)
            if path.exists():
                verify_packet(path, index, configuration_hash)
                digest = sha256_file(path)
                if key in run["prediction_sha256"] and run["prediction_sha256"][key] != digest:
                    raise ValueError("Existing prediction content hash differs")
                reused += 1
            else:
                rgb = np.ascontiguousarray(dataset["images"][index].transpose(2, 1, 0))
                values = reference.predict_pair(rgb)
                values.update({f"{name}_m": predictor.predict_rgb(rgb).values for name, predictor in predictors.items()})
                if any(value.shape != (480, 640) or not np.isfinite(value).all() for value in values.values()):
                    raise ValueError("Research prediction dimensions or values differ")
                temporary = path.with_suffix(".pending.npz")
                np.savez_compressed(temporary, image_id=index, configuration_sha256=configuration_hash, **values)
                replace_file(temporary, path)
                digest = sha256_file(path)
                written += 1
            run["prediction_sha256"][key] = digest
            run["completed_predictions"] = number + 1
            write_json(run_path, run)
            if (number + 1) % 50 == 0 or number + 1 == len(config["test_ids"]):
                print(f"Frozen research predictions {number + 1}/{len(config['test_ids'])}", flush=True)
    if cuda:
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start
    result = evaluate_metric_packets(root, config_path, output)
    hardware = {"device": predictors["selected"].device, "precision": predictors["selected"].metadata["precision"],
                "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda}
    if cuda:
        hardware["gpu"] = torch.cuda.get_device_name()
    summary = {**result, "config": config, "provenance": run["provenance"], "hardware": hardware,
               "prediction_files_sha256": run["prediction_sha256"],
               "generation": {"last_invocation_s": generation_seconds, "written_packets": written, "reused_packets": reused,
                              "scope": "Current invocation with warmups, compared pipelines, transfers and compressed packet I/O"}}
    write_json(output / "summary.json", summary)
    return summary
