import numpy as np
import pytest
import torch

from depth_project import metric_research as research


def test_nested_subsets_keep_whole_scenes_and_source_ids():
    scenes = {index: f"room-{index // 3}" for index in range(18)}
    subsets = research.nested_scene_subsets(scenes, [4, 9, 18], seed=7)
    assert subsets == research.nested_scene_subsets(scenes, [4, 9, 18], seed=7)
    assert set(subsets[4]) < set(subsets[9]) < set(subsets[18])
    assert len(subsets[4]) >= 4
    for ids in subsets.values():
        chosen_scenes = {scenes[index] for index in ids}
        assert set(ids) == {index for index, scene in scenes.items() if scene in chosen_scenes}
    assert len(subsets[18]) == 18


def test_scene_folds_are_disjoint_and_cover_fitting_pool():
    scenes = {index: f"room-{index // 2}" for index in range(12)}
    folds = research.scene_folds(scenes, count=3, seed=7)
    assert sorted(index for fold in folds for index in fold) == list(range(12))
    for first, second in zip(folds, folds[1:]):
        assert {scenes[index] for index in first}.isdisjoint(scenes[index] for index in second)


def test_partition_validation_rejects_scene_and_test_overlap():
    scenes = {0: "a", 1: "a", 2: "b", 3: "c"}
    research.validate_partition([0, 1], [2], [3], scenes)
    with pytest.raises(ValueError, match="scene"):
        research.validate_partition([0], [1], [3], scenes)
    with pytest.raises(ValueError, match="overlap"):
        research.validate_partition([0, 1], [2], [1, 3], scenes)
    with pytest.raises(ValueError, match="duplicate"):
        research.validate_partition([0, 0], [2], [3], scenes)


def test_calibration_uses_only_declared_fitting_labels():
    predictions = {index: np.ones((480, 640), np.float32) for index in range(3)}
    truths = {0: predictions[0] * 2, 1: predictions[1] * 2, 2: predictions[2] * 100}
    scale = research.fit_subset_scale(predictions, truths, [0, 1])
    truths[2][:] = 0.5
    assert scale == pytest.approx(2)
    assert research.fit_subset_scale(predictions, truths, [0, 1]) == scale


def test_boundaries_ignore_invalid_neighbors_and_crop_exterior():
    truth = np.ones((480, 640), np.float32)
    truth[:, 320:] = 2
    truth[120:140, 100:120] = np.nan
    band = research.depth_boundary_band(truth)
    assert band[200, 319:322].all()
    assert not band[:45].any()
    assert not band[:, :41].any()
    assert not band[115:145, 95:125].any()


def test_torch_metrics_match_course_evaluator():
    from depth_project.metrics import evaluate_nyu

    generator = np.random.default_rng(42)
    truth = generator.uniform(0.1, 11, (480, 640)).astype(np.float32)
    truth[100:110, 100:110] = np.nan
    prediction = generator.uniform(0, 12, (480, 640)).astype(np.float32)
    expected = evaluate_nyu(prediction, truth)
    actual = research.torch_nyu_metrics(torch.from_numpy(prediction[None, None]), torch.from_numpy(truth[None, None]))
    for index, name in enumerate(["abs_rel", "rmse", "delta1"]):
        assert actual[0, index].item() == pytest.approx(expected[name], abs=1e-10)


def test_gradient_loss_masks_invalid_pairs_and_matches_constant_truth():
    truth = torch.ones(1, 1, 12, 16)
    prediction = truth.clone().requires_grad_()
    mask = torch.ones_like(truth, dtype=torch.bool)
    assert research.log_gradient_loss(prediction, truth, mask).item() == 0
    truth[:, :, 3:5, 3:5] = float("nan")
    mask &= torch.isfinite(truth)
    loss = research.log_gradient_loss(prediction, truth, mask)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_refit_requires_complete_development_and_fixed_duration():
    research.validate_refit([0, 1, 2], [0, 1, 2], [3], epochs=20, validation_ids=[])
    with pytest.raises(ValueError, match="development"):
        research.validate_refit([0, 1], [0, 1, 2], [3], epochs=20, validation_ids=[])
    with pytest.raises(ValueError, match="validation"):
        research.validate_refit([0, 1, 2], [0, 1, 2], [3], epochs=20, validation_ids=[2])
    with pytest.raises(ValueError, match="duration"):
        research.validate_refit([0, 1, 2], [0, 1, 2], [3], epochs=0, validation_ids=[])


def test_partition_requires_scene_identity_for_every_id():
    with pytest.raises(ValueError, match="missing"):
        research.validate_partition([0], [1], [2], {0: "a", 1: "b"})


def test_torch_metrics_reject_nonfinite_predictions():
    prediction = torch.ones(1, 1, 480, 640)
    prediction[..., 100, 100] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        research.torch_nyu_metrics(prediction, torch.ones_like(prediction))
