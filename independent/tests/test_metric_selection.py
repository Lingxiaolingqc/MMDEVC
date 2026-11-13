import numpy as np
import pytest

from depth_project.metric_selection import fixed_refit_epochs, select_seed_ensemble, paired_scene_comparison


def test_refit_duration_is_fixed_from_three_fold_checkpoints():
    assert fixed_refit_epochs([3, 9, 5]) == 5
    assert fixed_refit_epochs([2.5, 5.5, 8.5]) == 6
    with pytest.raises(ValueError):
        fixed_refit_epochs([3, np.nan, 5])


def test_equal_ensemble_is_compared_with_the_median_single_seed():
    truth = np.full((1, 480, 640), 2.0, dtype=np.float32)
    predictions = {7: truth * 0.75, 8: truth.copy(), 9: truth * 1.5}
    selected = select_seed_ensemble(predictions, truth)
    assert selected["median_seed"] == 7
    assert selected["chosen_seeds"] == [7, 8, 9]
    assert selected["ensemble_abs_rel"] < selected["median_single_abs_rel"]


def test_single_seed_wins_an_exact_ensemble_tie_and_invalid_predictions_fail():
    truth = np.full((1, 480, 640), 2.0, dtype=np.float32)
    selected = select_seed_ensemble({9: truth, 7: truth, 8: truth}, truth)
    assert selected["chosen_seeds"] == [7]
    assert selected["median_seed"] == 7
    with pytest.raises(ValueError, match="positive"):
        select_seed_ensemble({7: truth, 8: truth, 9: truth * 0}, truth)


def test_scene_comparison_pairs_ids_and_reports_known_constant_gain():
    before = [{"image_id": i, "scene": str(i // 2), "abs_rel": 0.2} for i in range(6)]
    after = [{**row, "abs_rel": 0.1} for row in before]
    result = paired_scene_comparison(before, after)
    assert result["relative_abs_rel_reduction_percent"] == pytest.approx(50)
    assert result["scene_bootstrap_95_interval_percent"] == pytest.approx([50, 50])
    assert result["improved_images"] == 6
    with pytest.raises(ValueError, match="paired"):
        paired_scene_comparison(before, after[::-1])
