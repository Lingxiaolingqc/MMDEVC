"""Shared validation and previews with the parent's numerical scoring semantics."""
import time
import numpy as np
import torch
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.model import corrected_depth
from p3_depth.residual.evaluation import score_image


def batch(pack, features, internal, base, positions):
    def tensor(value):
        return torch.from_numpy(np.ascontiguousarray(value[positions])).to('cuda')
    return (tensor(features), {key:tensor(value).float() for key,value in internal.items()},
            tensor(base)[:, None], tensor(pack['truth'])[:, None], tensor(pack['mask'])[:, None], tensor(pack['band'])[:, None])


def evaluate(model, config, pack, features, internal, base, positions, batch_size, detailed=False):
    model.eval()
    rows = []
    times = []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            selected = positions[start:start+batch_size]
            inputs, latent, bases, truths, masks, _ = batch(pack, features, internal, base, selected)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                residual = model(inputs, latent)
            prediction = corrected_depth(bases, residual, config['output'])
            torch.cuda.synchronize()
            times.append((time.perf_counter()-tick, len(selected)))
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError('Nonfinite validation prediction')
            if detailed:
                array = prediction[:, 0].cpu().numpy()
                rows.extend(score_image(array[j], pack, i) for j, i in enumerate(selected))
            else:
                clipped, truth = prediction.clamp(.001, 10).double(), truths.double()
                safe = torch.where(masks, truth, 1.)
                count = masks.sum((1, 2, 3))
                difference = clipped - safe
                scores = torch.stack([((difference.abs()/safe)*masks).sum((1, 2, 3))/count,
                                      ((difference.square()*masks).sum((1, 2, 3))/count).sqrt(),
                                      ((torch.maximum(clipped/safe, safe/clipped) < 1.25)*masks).sum((1, 2, 3))/count], -1).cpu().numpy()
                rows.extend({'abs_rel': float(v[0]), 'rmse_m': float(v[1]), 'delta1': float(v[2])} for v in scores)
    mean = {key: float(np.mean([r[key] for r in rows])) for key in ('abs_rel', 'rmse_m', 'delta1')}
    mean['refiner_seconds_per_image'] = sum(x[0] for x in times)/sum(x[1] for x in times)
    return mean, rows


def preview(root, study, output, model, config, pack, features, internal, base, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed before looking at candidate scores: first/middle/last validation samples.
    selected = [positions[i] for i in sorted({0, len(positions)//2, len(positions)-1})]
    with NYUv2Labeled(root / study['data']) as data, torch.inference_mode():
        for i in selected:
            inputs, latent, bases, _, _, _ = batch(pack, features, internal, base, [i])
            with torch.autocast('cuda', dtype=torch.float16):
                residual = model(inputs, latent)
            predicted = corrected_depth(bases, residual, config['output'])[0, 0].cpu().numpy()
            rgb = data.load(pack['ids'][i]+1).rgb
            fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), constrained_layout=True)
            axes[0].imshow(rgb)
            for ax, values in zip(axes[1:4], [pack['truth'][i], base[i], predicted]):
                ax.imshow(values, vmin=0, vmax=10, cmap='viridis')
            difference = predicted - base[i]
            axes[4].imshow(difference, vmin=-.5, vmax=.5, cmap='coolwarm')
            for ax, title in zip(axes, ['RGB', 'GT (0-10m)', 'Base (0-10m)', 'Refined (0-10m)', 'Correction (-.5,+.5m)']):
                ax.set_title(title)
                ax.axis('off')
            fig.suptitle(f'{config["name"]} | nyuv2_{pack["ids"][i]+1:04d} | validation; fixed display limits')
            fig.savefig(output / f'nyuv2_{pack["ids"][i]+1:04d}.png', dpi=140)
            plt.close(fig)
