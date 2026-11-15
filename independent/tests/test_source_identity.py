from depth_project.records import sha256_source


def test_source_identity_survives_git_line_endings_and_detects_code_changes(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\nprint(value)\n")
    expected = sha256_source(path)
    path.write_bytes(b"value = 1\r\nprint(value)\r\n")
    assert sha256_source(path) == expected
    path.write_bytes(b"value = 2\r\nprint(value)\r\n")
    assert sha256_source(path) != expected
