"""Reproducible small-refiner training with explicit development partitions."""
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import torch
from filelock import FileLock
from safetensors.torch import save_file
from torch.nn import functional as F

from .data import load_nyu_split
from .metric_research import fit_subset_scale, log_gradient_loss, torch_nyu_metrics, validate_partition, validate_refit
from .records import run_provenance, sha256_file
from .refinement import blend_flip
from .residual_refiner import ResidualRefiner, make_features


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_development_pack(root):
    root = Path(root)
    manifest = json.loads((root / "runs/nyu-development/manifest.json").read_text())
    split = load_nyu_split(root / "data/nyu/splits.mat")
    identities = manifest["identities"]
    for name, path in [("dataset_sha256", "data/nyu/nyu_depth_v2_labeled.mat"), ("split_sha256", "data/nyu/splits.mat")]:
        if sha256_file(root / path) != identities[name]:
            raise ValueError("Development source identity differs from the recorded cache")
    ids = sorted(manifest["fit_ids"] + manifest["validation_ids"])
    if ids != sorted(split["development"].tolist()):
        raise ValueError("Development manifest does not cover the official split")
    prepared = root / ".cache/metric-research/development.pt"
    if prepared.exists():
        stored = torch.load(prepared, weights_only=True, mmap=True)
        if stored["identities"] != identities or stored["ids"] != ids:
            raise ValueError("Prepared development cache identity differs")
        return {**stored, **{key: stored[key].numpy() for key in ["features", "base", "truth"]}}
    features = np.empty((len(ids), 9, 240, 320), np.float16)
    base = np.empty((len(ids), 480, 640), np.float32)
    truth = np.empty_like(base)
    scenes = {}
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as source:
        for index in range(source["images"].shape[0]):
            scenes[index] = "".join(chr(int(value)) for value in source[source["scenes"][0, index]][:].reshape(-1))
        validate_partition(manifest["fit_ids"], manifest["validation_ids"], split["test"].tolist(), scenes)
        for position, index in enumerate(ids):
            path = root / f"runs/nyu-development/prediction_{index:04d}.npz"
            with np.load(path, allow_pickle=False) as prediction:
                if int(prediction["image_index"]) != index or str(prediction["checkpoint_sha256"]) != identities["checkpoint_sha256"]:
                    raise ValueError("Prediction cache identity differs")
                original, flipped, confidence = [prediction[key] for key in ["depth_m", "flip_m", "confidence"]]
                if any(values.shape != (480, 640) or not np.isfinite(values).all() for values in [original, flipped, confidence]):
                    raise ValueError("Prediction cache contains invalid tensors")
                base[position] = blend_flip(original, flipped, 0.25)
                rgb = np.ascontiguousarray(source["images"][index].transpose(2, 1, 0))
                features[position] = make_features(rgb, original, flipped, confidence, base[position])
                truth[position] = source["depths"][index].T
            if (position + 1) % 100 == 0:
                print(f"Preparing development tensors {position + 1}/{len(ids)}", flush=True)
    pack = {"ids": ids, "scenes": scenes, "base": base, "truth": truth, "features": features,
            "fit_ids": manifest["fit_ids"], "validation_ids": manifest["validation_ids"],
            "test_ids": split["test"].tolist(), "identities": identities}
    prepared.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.with_suffix(".pending.pt")
    torch.save({**pack, **{key: torch.from_numpy(pack[key]) for key in ["features", "base", "truth"]}}, temporary)
    temporary.replace(prepared)
    return pack


def scaled_features(features, base, scale):
    result = np.asarray(features, dtype=np.float32).copy()
    height, width = result.shape[-2:]
    for position, values in enumerate(base):
        scaled = np.maximum(values * scale, 0.001)
        low = cv2.resize(scaled, (width, height), interpolation=cv2.INTER_AREA)
        result[position, 5] = np.log(np.maximum(low, 0.001)) / 2
    return result


def augment_refiner_rgb(features):
    result = features.clone()
    image = (result[:, :3] + 1) / 2
    factors = 0.9 + 0.2 * torch.rand((3, len(image), 1, 1, 1), device=image.device)
    image = image * factors[0]
    image = (image - image.mean(dim=(1, 2, 3), keepdim=True)) * factors[1] + image.mean(dim=(1, 2, 3), keepdim=True)
    luminance = 0.299 * image[:, :1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    image = luminance + factors[2] * (image - luminance)
    result[:, :3] = image.clamp(0, 1) * 2 - 1
    return result


def train_candidate(pack, config, output, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        return _train_candidate(pack, config, output, device)


def _train_candidate(pack, config, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity_payload = {"config": config, "data": pack["identities"]}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["configuration_sha256"] != identity:
            raise ValueError("Completed training configuration identity differs")
        if sha256_file(output / "best.safetensors") != summary["checkpoint_sha256"]:
            raise ValueError("Completed training checkpoint identity differs")
        return summary
    run_path = output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["configuration_sha256"] != identity:
        raise ValueError("Incomplete training configuration identity differs")
    fitting_ids = list(config["fit_ids"])
    validation_ids = list(config["validation_ids"])
    validate_partition(fitting_ids, validation_ids, pack["test_ids"], pack["scenes"])
    if not validation_ids:
        validate_refit(fitting_ids, pack["ids"], pack["test_ids"], config["max_epochs"], validation_ids)
    positions = {index: position for position, index in enumerate(pack["ids"])}
    fitting = [positions[index] for index in fitting_ids]
    validation = [positions[index] for index in validation_ids]
    predictions = {index: pack["base"][positions[index]] for index in fitting_ids}
    truths = {index: pack["truth"][positions[index]] for index in fitting_ids}
    scale = fit_subset_scale(predictions, truths, fitting_ids)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = device == "cuda"
    feature_array = scaled_features(pack["features"], pack["base"], scale).astype(np.float16)
    features = torch.from_numpy(feature_array).to(device)
    base_array = np.maximum(pack["base"] * scale, 0.001)
    bases = torch.from_numpy(base_array[:, None]).to(device)
    full_truth = torch.from_numpy(np.asarray(pack["truth"])[:, None]).to(device)
    height, width = features.shape[-2:]
    low_bases = torch.from_numpy(np.stack([cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR) for values in base_array])[:, None]).to(device)
    low_truth = torch.from_numpy(np.stack([cv2.resize(values, (width, height), interpolation=cv2.INTER_NEAREST) for values in pack["truth"]])[:, None]).to(device)
    full_mask = torch.zeros_like(full_truth, dtype=torch.bool)
    full_mask[:, :, 45:471, 41:601] = True
    full_mask &= torch.isfinite(full_truth) & (full_truth > 0.001) & (full_truth < 10)
    low_mask = torch.zeros_like(low_truth, dtype=torch.bool)
    low_mask[:, :, math.ceil(45 * height / 480):math.floor(471 * height / 480), math.ceil(41 * width / 640):math.floor(601 * width / 640)] = True
    low_mask &= torch.isfinite(low_truth) & (low_truth > 0.001) & (low_truth < 10)
    frozen = None
    if config.get("model", {}).get("feature_channels"):
        frozen = torch.from_numpy(pack["depth_features"]).to(device)
    model = ResidualRefiner(**config.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4), weight_decay=config.get("weight_decay", 0.001))
    batch_size = config.get("batch_size", 8)
    steps_per_epoch = math.ceil(len(fitting) / batch_size)
    maximum_steps = config.get("max_steps") or config.get("max_epochs", 80) * steps_per_epoch
    schedule_steps = config.get("max_steps") or config.get("schedule_epochs", 80) * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    fitting_tensor = torch.tensor(fitting, device=device)
    history = []
    best_score = float("inf")
    best_step = 0
    step = 0
    exposures = 0
    cursor = len(fitting)
    order = fitting_tensor
    start = time.perf_counter()

    def evaluate(indices):
        model.eval()
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(indices), 8):
                selected = indices[offset:offset + 8]
                with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                    residual = model(features[selected].float(), frozen[selected] if frozen is not None else None)
                resized = F.interpolate(residual.float(), size=(480, 640), mode="bilinear", align_corners=False)
                scores.append(torch_nyu_metrics(bases[selected] * resized.exp(), full_truth[selected]).cpu())
        return torch.cat(scores).numpy()

    def save_best():
        temporary = output / "best.pending.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(temporary))
        temporary.replace(output / "best.safetensors")

    latest = output / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, exposures, cursor = state["step"], state["exposures"], state["cursor"]
        order = state["order"].to(device)
        best_score, best_step, history = state["best_score"], state["best_step"], state["history"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    else:
        initial_scores = evaluate(validation or fitting)
        best_score = float(initial_scores[:, 0].mean())
        save_best()
    if not run_path.exists():
        root = Path(__file__).resolve().parents[2]
        write_json(run_path, {"config": config, "configuration_sha256": identity, "data_identities": pack["identities"],
                              "scale": scale, "provenance": run_provenance(root)})
    losses = []
    while step < maximum_steps:
        if cursor >= len(fitting):
            order = fitting_tensor[torch.randperm(len(fitting), device=device)]
            cursor = 0
        selected = order[cursor:cursor + batch_size]
        cursor += len(selected)
        exposures += len(selected)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_features = features[selected].float()
        if config.get("augment_rgb"):
            batch_features = augment_refiner_rgb(batch_features)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            residual = model(batch_features, frozen[selected] if frozen is not None else None)
            if config.get("full_resolution"):
                residual = F.interpolate(residual.float(), size=(480, 640), mode="bilinear", align_corners=False)
                predicted = (bases[selected] * residual.exp()).clamp(0.001, 10)
                target, valid = full_truth[selected], full_mask[selected]
            else:
                predicted = (low_bases[selected] * residual.float().exp()).clamp(0.001, 10)
                target, valid = low_truth[selected], low_mask[selected]
            safe_target = torch.where(valid, target, 1)
            relative = torch.where(valid, (predicted - safe_target).abs() / safe_target, 0)
            loss = (relative.sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3))).mean()
            if config.get("gradient_weight", 0):
                loss = loss + config["gradient_weight"] * log_gradient_loss(predicted, target, valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Training produced a non-finite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        losses.append(float(loss.detach()))
        interval = config.get("evaluate_every", steps_per_epoch * 2)
        if step % interval and step != maximum_steps:
            continue
        scores = evaluate(validation or fitting)
        score = float(scores[:, 0].mean())
        if score < best_score or not validation:
            best_score, best_step = score, step
            save_best()
        row = {"step": step, "epoch": exposures / len(fitting), "optimization_loss": float(np.mean(losses)),
               "abs_rel": score, "rmse": float(scores[:, 1].mean()), "delta1": float(scores[:, 2].mean()),
               "best_step": best_step, "elapsed_s": time.perf_counter() - start}
        if config.get("measure_fitting", False):
            row["fitting_full_resolution_abs_rel"] = float(evaluate(fitting)[:, 0].mean())
        history.append(row)
        losses.clear()
        write_json(output / "progress.json", history)
        print(json.dumps({"run": output.name, **row}), flush=True)
        temporary = latest.with_suffix(".pending.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "exposures": exposures, "cursor": cursor, "order": order,
                    "best_score": best_score, "best_step": best_step, "history": history, "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else torch.empty(0, dtype=torch.uint8)}, temporary)
        temporary.replace(latest)
        if validation and step >= config.get("minimum_steps", steps_per_epoch * 24) and step - best_step >= config.get("patience", 10) * interval:
            break
    result = {"configuration_sha256": identity, "config": config, "fit_ids": fitting_ids, "validation_ids": validation_ids,
              "scale": scale, "completed_steps": step, "best_step": best_step, "best_epoch": best_step / steps_per_epoch,
              "best_abs_rel": best_score, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
              "history": history, "training_s": time.perf_counter() - start, "checkpoint_sha256": sha256_file(output / "best.safetensors"),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device == "cuda" else None}
    write_json(summary_path, result)
    return result
