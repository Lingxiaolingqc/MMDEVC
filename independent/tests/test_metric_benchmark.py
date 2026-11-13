from types import SimpleNamespace

import numpy as np
import pytest

from depth_project import benchmark


def test_metric_benchmark_runs_every_image_with_warmups_and_gpu_synchronization(monkeypatch):
    calls = []
    adapter = SimpleNamespace(device="cuda", predict_rgb=lambda rgb: calls.append(rgb))
    events = []
    cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: events.append("sync"),
                           reset_peak_memory_stats=lambda: events.append("reset"),
                           max_memory_allocated=lambda: 1024 ** 2, max_memory_reserved=lambda: 2 * 1024 ** 2)
    timer = iter(range(12))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(timer))
    images = [np.zeros((4, 5, 3), np.uint8), np.ones((4, 5, 3), np.uint8)]
    result = benchmark.benchmark_metric_images(images, adapter, repeats=3, torch_module=SimpleNamespace(cuda=cuda))
    assert len(calls) == 8
    assert result["median_image_s"] == 1
    assert result["peak_allocated_mib"] == 1
    assert result["peak_reserved_mib"] == 2
    assert events.count("reset") == 3
    assert events.count("sync") >= 12


def test_metric_benchmark_rejects_an_empty_image_sequence():
    with pytest.raises(ValueError):
        benchmark.benchmark_metric_images([], SimpleNamespace(device="cpu"))
