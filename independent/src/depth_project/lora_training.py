"""Scene-separated, resumable Q/V adaptation of the pretrained metric model."""
import hashlib
import json
from pathlib import Path
import time

from filelock import FileLock
import h5py
import numpy as np
from safetensors.torch import save_file
import torch

from .atomic_io import replace_file
from .metric_adaptation import configure_lora, infer_differentiable
from .metric_research import fit_subset_scale, torch_nyu_metrics, validate_partition, validate_refit
from .metric_training import write_json
from .models import load_image_adapter
from .records import run_provenance, sha256_file


def adapter_state(model):
    return {name: parameter.detach().cpu().clone().contiguous()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_adapter_state(model, state):
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("Adapter weight names differ from the configured architecture")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape or not torch.isfinite(state[name]).all():
                raise ValueError("Adapter weight shape or values differ")
            parameter.copy_(state[name])


def frozen_digest(model):
    digest = hashlib.sha256()
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    for name, values in frozen + list(model.named_buffers()):
        digest.update(name.encode())
        digest.update(values.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_lora_candidate(root, pack, config, output, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        return _train(root, pack, config, output, device)


def _train(root, pack, config, output, device):
    root = Path(root)
    identity = hashlib.sha256(json.dumps({"config": config, "data": pack["identities"]}, sort_keys=True).encode()).hexdigest()
    summary_path = output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["configuration_sha256"] != identity:
            raise ValueError("Completed LoRA configuration differs")
        if sha256_file(output / "best.safetensors") != previous["checkpoint_sha256"]:
            raise ValueError("Completed LoRA checkpoint differs")
        return previous
    run_path = output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["configuration_sha256"] != identity:
        raise ValueError("Incomplete LoRA configuration differs")
    fitting, validation = list(config["fit_ids"]), list(config["validation_ids"])
    validate_partition(fitting, validation, pack["test_ids"], pack["scenes"])
    if not validation:
        validate_refit(fitting, pack["ids"], pack["test_ids"], config["max_epochs"], validation)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    adapter = load_image_adapter("unidepth", root)
    if adapter.metadata["checkpoint_sha256"] != pack["identities"]["checkpoint_sha256"]:
        raise ValueError("LoRA foundation checkpoint differs")
    model = adapter.model
    positions = {index: position for position, index in enumerate(pack["ids"])}
    original = {}
    for index in fitting:
        with np.load(root / f"runs/nyu-development/prediction_{index:04d}.npz", allow_pickle=False) as packet:
            original[index] = packet["depth_m"]
    scale = fit_subset_scale(original, {index: pack["truth"][positions[index]] for index in fitting}, fitting)
    expected = torch.from_numpy(original[fitting[0]][None, None]).to(device)
    del original
    names = configure_lora(model, config["rank"])
    frozen_before = frozen_digest(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.get("schedule_epochs", 80), eta_min=1e-5)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    truth = {index: torch.from_numpy(pack["truth"][positions[index]][None, None]).to(device) for index in fitting + validation}
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        images = {index: torch.from_numpy(np.ascontiguousarray(dataset["images"][index].transpose(0, 2, 1))).to(device)
                  for index in fitting + validation}

    def predict(index):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            return infer_differentiable(model, images[index]) * scale

    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        initial = infer_differentiable(model, images[fitting[0]])
    difference = float((initial - expected).abs().max())
    if difference != 0:
        raise ValueError(f"Zero-update LoRA differs from the recorded reference by {difference}")

    def evaluate(indices, collect=False):
        rows, predictions = [], []
        with torch.no_grad():
            for index in indices:
                prediction = predict(index)
                scores = torch_nyu_metrics(prediction, truth[index])[0].cpu().tolist()
                rows.append({"image_id": index, "scene": pack["scenes"][index], "abs_rel": scores[0],
                             "rmse": scores[1], "delta1": scores[2]})
                if collect:
                    predictions.append(prediction[0, 0].float().cpu().numpy())
        return rows, predictions

    def save_best(state):
        temporary = output / "best.pending.safetensors"
        save_file(state, str(temporary))
        replace_file(temporary, output / "best.safetensors")

    latest = output / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=True)
        if state["configuration_sha256"] != identity:
            raise ValueError("Resume configuration differs")
        load_adapter_state(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, completed = state["history"], state["epoch"]
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        best_state = {key: value.cpu().contiguous() for key, value in state["best_model"].items()}
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        save_best(best_state)
    else:
        rows, _ = evaluate(validation or fitting)
        best_score = float(np.mean([row["abs_rel"] for row in rows]))
        best_epoch, completed = 0, 0
        best_state = adapter_state(model)
        save_best(best_state)
        history = [{"epoch": 0, "abs_rel": best_score, "elapsed_s": 0.0}]
    if not run_path.exists():
        provenance = run_provenance(root)
        provenance["randomness"] = {"seed": seed, "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        write_json(run_path, {"config": config, "configuration_sha256": identity, "data_identities": pack["identities"],
                              "scale": scale, "trainable_names": names, "provenance": provenance})
    start = time.perf_counter()
    elapsed_before = history[-1]["elapsed_s"]
    patience = config.get("patience", 20)
    minimum_epochs = config.get("minimum_epochs", 24)
    accumulation = config.get("accumulation", 8)
    stopped = bool(validation and completed >= minimum_epochs and completed - best_epoch >= patience)
    for epoch in range(completed + 1, config["max_epochs"] + 1):
        if stopped:
            break
        order = np.random.default_rng(seed + epoch).permutation(fitting)
        losses = []
        for offset in range(0, len(order), accumulation):
            indices = order[offset:offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                index = int(index)
                prediction = predict(index).clamp(0.001, 10)[:, :, 45:471, 41:601]
                target = truth[index][:, :, 45:471, 41:601]
                valid = torch.isfinite(target) & (target > 0.001) & (target < 10)
                denominator = torch.where(valid, target, 1)
                errors = torch.where(valid, (prediction - denominator).abs() / denominator, 0)
                loss = errors.sum() / valid.sum()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("LoRA produced a non-finite training loss")
                scaler.scale(loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise ValueError("LoRA gradients are missing or non-finite")
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        rows, _ = evaluate(validation or fitting)
        score = float(np.mean([row["abs_rel"] for row in rows]))
        if score < best_score or not validation:
            best_score, best_epoch = score, epoch
            best_state = adapter_state(model)
            save_best(best_state)
        row = {"epoch": epoch, "optimization_abs_rel": float(np.mean(losses)), "abs_rel": score,
               "rmse": float(np.mean([item["rmse"] for item in rows])),
               "delta1": float(np.mean([item["delta1"] for item in rows])), "best_epoch": best_epoch,
               "elapsed_s": elapsed_before + time.perf_counter() - start}
        history.append(row)
        completed = epoch
        write_json(output / "progress.json", history)
        print(json.dumps({"run": output.name, **row}), flush=True)
        temporary = latest.with_suffix(".pending.pt")
        torch.save({"configuration_sha256": identity, "model": adapter_state(model), "best_model": best_state,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": completed, "history": history, "best_score": best_score, "best_epoch": best_epoch,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else torch.empty(0, dtype=torch.uint8)}, temporary)
        replace_file(temporary, latest)
        stopped = bool(validation and completed >= minimum_epochs and completed - best_epoch >= patience)
    if frozen_digest(model) != frozen_before:
        raise ValueError("Frozen foundation tensors changed during LoRA training")
    load_adapter_state(model, best_state)
    rows, predictions = evaluate(validation or fitting, collect=bool(validation))
    if abs(np.mean([row["abs_rel"] for row in rows]) - best_score) > 1e-7:
        raise ValueError("Reloaded LoRA checkpoint does not reproduce its score")
    if validation:
        temporary = output / "validation.pending.npz"
        np.savez_compressed(temporary, image_ids=np.array(validation), depth_m=np.stack(predictions))
        replace_file(temporary, output / "validation.npz")
    result = {"config": config, "configuration_sha256": identity, "scale": scale, "best_abs_rel": best_score,
              "best_epoch": best_epoch, "completed_epochs": completed, "history": history, "rows": rows,
              "score_scope": "validation" if validation else "fitting", "trainable_names": names,
              "trainable_parameters": sum(parameter.numel() for parameter in parameters),
              "frozen_parameters_unchanged": True, "initial_maximum_baseline_difference_m": difference,
              "checkpoint_sha256": sha256_file(output / "best.safetensors"),
              "training_s": elapsed_before + time.perf_counter() - start,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device == "cuda" else None}
    write_json(summary_path, result)
    return result
