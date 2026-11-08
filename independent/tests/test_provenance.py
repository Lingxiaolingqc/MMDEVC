from depth_project import records


def test_source_edit_changes_recorded_provenance_without_requiring_git(tmp_path):
    source=tmp_path/'src/depth_project/example.py';source.parent.mkdir(parents=True)
    source.write_text('first')
    before=records.run_provenance(tmp_path)
    source.write_text('second')
    after=records.run_provenance(tmp_path)
    assert before['source_sha256']['src/depth_project/example.py']!=after['source_sha256']['src/depth_project/example.py']
    assert before['python_version']
