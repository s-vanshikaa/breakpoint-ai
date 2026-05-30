from runner import checkpoint
from tests.conftest import make_record


def records(*ids):
    return [make_record(i, "direct_injection", True) for i in ids]


class TestAppendAndLoad:
    def test_load_missing_checkpoint_is_empty(self, tmp_path):
        assert checkpoint.load_records(tmp_path / "nope.jsonl") == []

    def test_appended_records_load_back_in_order(self, tmp_path):
        path = checkpoint.checkpoint_path(tmp_path, "baseline", 1)
        for r in records("a", "b", "c"):
            checkpoint.append_record(path, r)
        loaded = checkpoint.load_records(path)
        assert [r.test_id for r in loaded] == ["a", "b", "c"]

    def test_append_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "trials" / "baseline-seed-1.checkpoint.jsonl"
        checkpoint.append_record(path, records("a")[0])
        assert path.is_file()

    def test_truncated_trailing_line_is_skipped_not_raised(self, tmp_path):
        path = checkpoint.checkpoint_path(tmp_path, "baseline", 1)
        for r in records("a", "b"):
            checkpoint.append_record(path, r)
        with path.open("a") as f:
            f.write('{"test_id": "c", "broken')  # simulates a crash mid-write

        loaded = checkpoint.load_records(path)
        assert [r.test_id for r in loaded] == ["a", "b"]

    def test_blank_lines_are_ignored(self, tmp_path):
        path = checkpoint.checkpoint_path(tmp_path, "baseline", 1)
        checkpoint.append_record(path, records("a")[0])
        with path.open("a") as f:
            f.write("\n\n")
        checkpoint.append_record(path, records("b")[0])
        loaded = checkpoint.load_records(path)
        assert [r.test_id for r in loaded] == ["a", "b"]


class TestClear:
    def test_clear_removes_the_file(self, tmp_path):
        path = checkpoint.checkpoint_path(tmp_path, "baseline", 1)
        checkpoint.append_record(path, records("a")[0])
        checkpoint.clear(path)
        assert not path.exists()

    def test_clear_missing_file_does_not_raise(self, tmp_path):
        checkpoint.clear(tmp_path / "nope.jsonl")


class TestPathNaming:
    def test_path_matches_trial_filename_convention(self, tmp_path):
        path = checkpoint.checkpoint_path(tmp_path, "guarded", 3)
        assert path.name == "guarded-seed-3.checkpoint.jsonl"
