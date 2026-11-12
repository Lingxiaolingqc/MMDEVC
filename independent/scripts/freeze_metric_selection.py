"""Freeze the development-selected metric method before test evaluation."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

from depth_project.records import sha256_file


def main():
    root = Path(__file__).resolve().parents[1]
    training = json.loads((root / 'runs/metric-refiner/training.json').read_text())
    search = json.loads((root / 'results/metric-development-search.json').read_text())
    output = root / 'configs/metric-final.json'
    if output.exists():
        raise FileExistsError('Metric selection already frozen')
    assert training['best_validation_abs_rel'] <= search['best_simple']['abs_rel']
    weights = root / 'runs/metric-refiner/best.pt'
    artifact = root / 'artifacts/metric-refiner.safetensors'
    artifact.parent.mkdir(exist_ok=True)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    save_file(state, str(artifact), metadata={'setting': 'NYU supervised development adaptation', 'seed': str(training['seed']), 'selected_epoch': str(training['best_epoch'])})
    config = {
        'method': 'scene-adapted RGB-guided residual refinement',
        'selection_status': 'frozen on scene-held-out development before final test',
        'foundation_checkpoint_sha256': search['identities']['checkpoint_sha256'],
        'dataset_sha256': search['identities']['dataset_sha256'],
        'split_sha256': search['identities']['split_sha256'],
        'refiner_checkpoint': 'artifacts/metric-refiner.safetensors',
        'refiner_sha256': sha256_file(artifact),
        'simple_config': training['base_config'],
        'original_global_scale': search['global_scales']['0'],
        'feature_hw': training['feature_hw'],
        'fit_ids': training['fit_ids'],
        'validation_ids': training['validation_ids'],
        'validation_baseline_abs_rel': search['rows'][0]['abs_rel'],
        'validation_selected_abs_rel': training['best_validation_abs_rel'],
        'selected_epoch': training['best_epoch'],
        'refiner_parameter_count': training['parameter_count'],
        'selection_metric': 'lowest full-resolution course-protocol AbsRel on 218 reserved development images from 62 scenes',
        'adaptation': '577 NYU development images used for fitting; 218 reserved for selection; foundation weights frozen; RGB-only test inference',
    }
    output.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({key: config[key] for key in ['method', 'selected_epoch', 'validation_baseline_abs_rel', 'validation_selected_abs_rel', 'refiner_sha256']}, indent=2))


if __name__ == '__main__':
    main()
