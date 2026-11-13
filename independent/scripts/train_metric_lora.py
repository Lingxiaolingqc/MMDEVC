"""Pilot Q/V-only adaptation of the pinned Small metric model."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from safetensors.torch import save_file

from depth_project.metric_adaptation import configure_lora, infer_differentiable
from depth_project.metric_research import fit_subset_scale, torch_nyu_metrics, validate_partition
from depth_project.metric_training import load_development_pack, write_json
from depth_project.models import load_image_adapter
from depth_project.records import run_provenance, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=[4, 8], default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--shared-gpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / f"runs/metric-research/lora-r{args.rank}-lr{args.learning_rate:g}-epochs{args.epochs}"
    output.mkdir(exist_ok=True)
    if (output / "summary.json").exists():
        print((output / "summary.json").read_text())
        return
    torch.set_num_threads(4)
    torch.manual_seed(320112)
    if args.shared_gpu:
        torch.cuda.set_per_process_memory_fraction(0.7)
    pack = load_development_pack(root)
    fitting, validation = pack["fit_ids"], pack["validation_ids"]
    validate_partition(fitting, validation, pack["test_ids"], pack["scenes"])
    adapter = load_image_adapter("unidepth", root)
    model = adapter.model
    positions = {index: position for position, index in enumerate(pack["ids"])}
    original = {}
    for index in fitting:
        with np.load(root / f"runs/nyu-development/prediction_{index:04d}.npz") as packet:
            original[index] = packet["depth_m"]
    scale = fit_subset_scale(original, {index: pack["truth"][positions[index]] for index in fitting}, fitting)
    del original
    names = configure_lora(model, args.rank)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    config = {"type": "lora", "rank": args.rank, "epochs": args.epochs, "learning_rate": args.learning_rate,
              "fit_ids": fitting, "validation_ids": validation, "scale": scale, "seed": 320112,
              "trainable_names": names, "trainable_parameters": sum(parameter.numel() for parameter in parameters),
              "foundation_sha256": adapter.metadata["checkpoint_sha256"], "provenance": run_provenance(root),
              "concurrent_gpu_work": args.shared_gpu}
    config["provenance"]["randomness"] = {"seed": 320112, "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    truth = {index: torch.from_numpy(pack["truth"][positions[index]][None, None]).cuda() for index in fitting + validation}
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        images = {index: torch.from_numpy(np.ascontiguousarray(dataset["images"][index].transpose(0, 2, 1))).cuda() for index in fitting + validation}

    def predict(index):
        with torch.autocast("cuda", dtype=torch.float16):
            return infer_differentiable(model, images[index]) * scale

    with np.load(root / f"runs/nyu-development/prediction_{fitting[0]:04d}.npz") as packet:
        expected = torch.from_numpy(packet["depth_m"][None, None]).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        initial = infer_differentiable(model, images[fitting[0]])
    difference = float((initial - expected).abs().max())
    if difference != 0:
        raise ValueError(f"Zero-update LoRA path differs from the recorded baseline by {difference}")
    config["initial_maximum_baseline_difference_m"] = difference
    write_json(output / "config.json", config)

    def evaluate():
        scores = []
        with torch.no_grad():
            for index in validation:
                scores.append(torch_nyu_metrics(predict(index), truth[index]).cpu())
        return torch.cat(scores).mean(dim=0).tolist()

    def save_adapter():
        temporary = output / "best.pending.safetensors"
        state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
        save_file(state, str(temporary))
        temporary.replace(output / "best.safetensors")

    scores = evaluate()
    best_score, best_epoch = scores[0], 0
    history = [{"epoch": 0, "abs_rel": scores[0], "rmse": scores[1], "delta1": scores[2]}]
    save_adapter()
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(320112 + epoch).permutation(fitting)
        losses = []
        for offset in range(0, len(order), 8):
            indices = order[offset:offset + 8]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                index = int(index)
                prediction = predict(index).clamp(0.001, 10)[:, :, 45:471, 41:601]
                target = truth[index][:, :, 45:471, 41:601]
                valid = torch.isfinite(target) & (target > 0.001) & (target < 10)
                denominator = torch.where(valid, target, 1)
                error = torch.where(valid, (prediction - denominator).abs() / denominator, 0)
                loss = error.sum() / valid.sum()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("LoRA adaptation produced non-finite loss")
                scaler.scale(loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        scores = evaluate()
        if scores[0] < best_score:
            best_score, best_epoch = scores[0], epoch
            save_adapter()
        row = {"epoch": epoch, "optimization_abs_rel": float(np.mean(losses)), "abs_rel": scores[0],
               "rmse": scores[1], "delta1": scores[2], "best_epoch": best_epoch, "elapsed_s": time.perf_counter() - start}
        history.append(row)
        write_json(output / "progress.json", history)
        print(json.dumps({"run": output.name, **row}), flush=True)
    summary = {"config": config, "best_abs_rel": best_score, "best_epoch": best_epoch, "history": history,
               "checkpoint_sha256": sha256_file(output / "best.safetensors"), "training_s": time.perf_counter() - start,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
