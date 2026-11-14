import pytest

from depth_project.metric_selection import rank_fold_results


def records(name, values):
    return [{"candidate": name, "fold": number,
             "metrics": {"abs_rel": value, "rmse": value * 3, "delta1": 0.8}}
            for number, value in enumerate(values)]


def test_fold_ranking_uses_all_three_fold_means():
    ranking = rank_fold_results(records("steady", [0.1, 0.1, 0.1]) + records("variable", [0.06, 0.06, 0.2]))
    assert ranking[0]["candidate"] == "steady"
    assert ranking[0]["mean_fold_metrics"]["abs_rel"] == pytest.approx(0.1)


def test_fold_ranking_rejects_missing_or_duplicate_folds():
    with pytest.raises(ValueError, match="three"):
        rank_fold_results(records("incomplete", [0.1, 0.2]))
    duplicate = records("duplicate", [0.1, 0.2, 0.3])
    duplicate[2]["fold"] = 1
    with pytest.raises(ValueError, match="three"):
        rank_fold_results(duplicate)
