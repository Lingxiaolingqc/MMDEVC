from pathlib import Path

import pytest

from depth_project import atomic_io


def test_atomic_replace_retries_transient_permission_errors(tmp_path, monkeypatch):
    source = tmp_path / "packet.pending.npz"
    source.write_bytes(b"complete packet")
    destination = tmp_path / "packet.npz"
    original = Path.replace
    attempts = []

    def held_open(path, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("another process holds the file")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", held_open)
    monkeypatch.setattr(atomic_io.time, "sleep", lambda duration: None)
    atomic_io.replace_file(source, destination)
    assert destination.read_bytes() == b"complete packet"
    assert len(attempts) == 3


def test_atomic_replace_propagates_persistent_failure(tmp_path, monkeypatch):
    def denied(path, target):
        raise PermissionError("persistent denial")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(atomic_io.time, "sleep", lambda duration: None)
    with pytest.raises(PermissionError, match="persistent"):
        atomic_io.replace_file(tmp_path / "source", tmp_path / "destination")
