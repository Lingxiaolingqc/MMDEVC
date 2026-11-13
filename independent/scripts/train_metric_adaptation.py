"""Evaluate lightweight adaptation of the existing UniDepth depth projections."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import save_file

from depth_project.metric_adaptation import ProjectionDepth
from depth_project.metric_research import torch_nyu_metrics, validate_partition
from depth_project.metric_training import load_development_pack, write_json
from depth_project.records import run_provenance, sha256_file


def prepare_projection_arrays(root, ids):
    source = root / "runs/metric-research/foundation-default"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("completed") != len(ids) or manifest["identities"]["ids"] != ids:
        raise ValueError("Complete the projection feature cache before training")
    folder = root / ".cache/metric-research/projection-arrays"
    folder.mkdir(exist_ok=True)
    shapes = {"projection": (len(ids), *manifest["projection_shape"]), "ray_z": (len(ids), 1, *manifest["processed_hw"])}
    completed = folder / "complete.json"
    if completed.exists():
        if json.loads(completed.read_text())["identities"] != manifest["identities"]:
            raise ValueError("Projection array cache identities differ")
    else:
        arrays = {name: np.lib.format.open_memmap(folder / f"{name}.pending.npy", mode="w+", dtype=np.float16 if name == "projection" else np.float32, shape=shape) for name, shape in shapes.items()}
        for position, index in enumerate(ids):
            with np.load(source / f"prediction_{index:04d}.npz", allow_pickle=False) as packet:
                if int(packet["image_id"]) != index:
                    raise ValueError("Projection packet image identity differs")
                arrays["projection"][position] = packet["projection_features"]
                arrays["ray_z"][position, 0] = packet["ray_z"]
            if (position + 1) % 100 == 0:
                print(f"Preparing projection arrays {position + 1}/{len(ids)}", flush=True)
        for values in arrays.values():
            values.flush()
        arrays.clear()
        del values
        for name in shapes:
            (folder / f"{name}.pending.npy").replace(folder / f"{name}.npy")
        write_json(completed, {"identities": manifest["identities"]})
    return {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in shapes}


def probe(root):
    source = root / "runs/metric-research/foundation-default"
    manifest = json.loads((source / "manifest.json").read_text())
    index = manifest["identities"]["ids"][0]
    model = ProjectionDepth().load_foundation(root / "checkpoints/unidepth-v2-vits14/model.safetensors").cuda()
    with np.load(source / f"prediction_{index:04d}.npz") as packet:
        features = torch.from_numpy(packet["projection_features"][None]).cuda()
        ray_z = torch.from_numpy(packet["ray_z"][None, None]).cuda()
        expected = torch.from_numpy(packet["depth_m"][None, None]).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        prediction = model(features, ray_z)
    difference = float((prediction - expected).abs().max().detach())
    if difference != 0:
        raise ValueError(f"Cached projection path differs from official depth by {difference}")
    prediction.mean().backward()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise ValueError("Projection gradient probe failed")
    record = {"image_id": index, "maximum_original_difference_m": difference,
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "all_projection_gradients_finite": True}
    write_json(root / "runs/metric-research/projection-probe.json", record)
    print(json.dumps(record), flush=True)


def train_projection(root, pack, arrays, learning_rate, epochs, batch_size):
    output = root / f"runs/metric-research/projection-lr{learning_rate:g}-epochs{epochs}-b{batch_size}"
    output.mkdir(exist_ok=True)
    fitting_ids = pack["fit_ids"]
    validation_ids = pack["validation_ids"]
    validate_partition(fitting_ids, validation_ids, pack["test_ids"], pack["scenes"])
    positions = {index: position for position, index in enumerate(pack["ids"])}
    fitting = [positions[index] for index in fitting_ids]
    validation = [positions[index] for index in validation_ids]
    torch.manual_seed(320112)
    model = ProjectionDepth().load_foundation(root / "checkpoints/unidepth-v2-vits14/model.safetensors").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    config = {"type": "projection", "learning_rate": learning_rate, "epochs": epochs,
              "fit_ids": fitting_ids, "validation_ids": validation_ids, "seed": 320112, "batch_size": batch_size,
              "foundation_sha256": pack["identities"]["checkpoint_sha256"]}
    if (output / "summary.json").exists():
        previous = json.loads((output / "summary.json").read_text())
        if previous["config"] != config or sha256_file(output / "best.safetensors") != previous["checkpoint_sha256"]:
            raise ValueError("Completed projection run identity differs")
        return previous
    write_json(output / "config.json", {**config, "provenance": run_provenance(root)})

    def predict(indices):
        features = torch.from_numpy(np.asarray(arrays["projection"][indices]).copy()).cuda()
        rays = torch.from_numpy(np.asarray(arrays["ray_z"][indices]).copy()).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            return model(features, rays)

    def evaluate():
        model.eval()
        values = []
        with torch.inference_mode():
            for offset in range(0, len(validation), batch_size):
                indices = validation[offset:offset + batch_size]
                truth = torch.from_numpy(pack["truth"][indices, None]).cuda()
                values.append(torch_nyu_metrics(predict(indices), truth).cpu())
        return torch.cat(values).mean(dim=0).tolist()

    def save_weights():
        temporary = output / "best.pending.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(temporary))
        temporary.replace(output / "best.safetensors")

    best = evaluate()
    best_epoch = 0
    save_weights()
    history = [{"epoch": 0, "abs_rel": best[0], "rmse": best[1], "delta1": best[2]}]
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(320112 + epoch).permutation(fitting)
        losses = []
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size].tolist()
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(indices).clamp(0.001, 10)
            truth = torch.from_numpy(pack["truth"][indices, None]).cuda()
            prediction = prediction[:, :, 45:471, 41:601]
            truth = truth[:, :, 45:471, 41:601]
            valid = torch.isfinite(truth) & (truth > 0.001) & (truth < 10)
            denominator = torch.where(valid, truth, 1)
            errors = torch.where(valid, (prediction - denominator).abs() / denominator, 0)
            loss = (errors.sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3))).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Projection adaptation produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        scores = evaluate()
        if scores[0] < best[0]:
            best, best_epoch = scores, epoch
            save_weights()
        row = {"epoch": epoch, "optimization_abs_rel": float(np.mean(losses)), "abs_rel": scores[0],
               "rmse": scores[1], "delta1": scores[2], "best_epoch": best_epoch, "elapsed_s": time.perf_counter() - start}
        history.append(row)
        write_json(output / "progress.json", history)
        print(json.dumps({"run": output.name, **row}), flush=True)
        if epoch >= 24 and epoch - best_epoch >= 20:
            break
    result = {"config": config, "best_abs_rel": best[0], "best_epoch": best_epoch, "history": history,
              "checkpoint_sha256": sha256_file(output / "best.safetensors"), "training_s": time.perf_counter() - start}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--shared-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    if args.probe:
        torch.cuda.set_per_process_memory_fraction(0.125)
        probe(root)
        return
    if args.shared_gpu:
        torch.cuda.set_per_process_memory_fraction(0.25)
    pack = load_development_pack(root)
    if sha256_file(root / "checkpoints/unidepth-v2-vits14/model.safetensors") != pack["identities"]["checkpoint_sha256"]:
        raise ValueError("Projection foundation checkpoint identity differs")
    arrays = prepare_projection_arrays(root, pack["ids"])
    for learning_rate in args.learning_rates or [args.learning_rate]:
        result = train_projection(root, pack, arrays, learning_rate, args.epochs, args.batch_size)
        print(json.dumps(result, indent=2), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
