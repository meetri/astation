"""Tests for the direct filesystem layer, with confinement first."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("trg_fs", PLUGIN_DIR / "sandbox_fs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trg_fs"] = mod
    spec.loader.exec_module(mod)
    return mod


fs = _load()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "sandbox").mkdir()
    return tmp_path / "sandbox"


def body(response) -> dict:
    return json.loads(response.content)


def test_a_parent_traversal_is_refused(root: Path):
    outside = root.parent / "secret.txt"
    outside.write_text("not yours")
    r = fs.read_text(str(root / ".." / "secret.txt"), str(root))
    assert r.status_code == 403
    assert "outside" in body(r)["detail"]


def test_an_absolute_path_outside_the_root_is_refused(root: Path):
    r = fs.read_text("/etc/passwd", str(root))
    assert r.status_code == 403


def test_a_symlink_pointing_out_of_the_root_is_refused(root: Path):
    """A lexical check alone would accept this: the path looks inside."""
    secret = root.parent / "secret.txt"
    secret.write_text("not yours")
    link = root / "innocent.txt"
    link.symlink_to(secret)
    r = fs.read_text(str(link), str(root))
    assert r.status_code == 403


def test_a_write_outside_the_root_is_refused_and_writes_nothing(root: Path):
    target = root.parent / "escape.txt"
    r = fs.write_text(str(target), "payload", str(root))
    assert r.status_code == 403
    assert not target.exists()


def test_mkdir_outside_the_root_is_refused(root: Path):
    target = root.parent / "escape-dir"
    r = fs.make_directory(str(target), str(root))
    assert r.status_code == 403
    assert not target.exists()


def test_the_root_itself_is_allowed(root: Path):
    r = fs.list_directory(str(root), str(root))
    assert r.status_code == 200


def test_read_text_returns_the_measured_shape(root: Path):
    (root / "a.py").write_text("print('hi')\n")
    r = fs.read_text(str(root / "a.py"), str(root))
    assert r.status_code == 200
    d = body(r)
    assert set(d) == {"binary", "byteSize", "language", "mimeType", "path", "text", "truncated"}
    assert d["text"] == "print('hi')\n"
    assert d["binary"] is False
    assert d["truncated"] is False
    assert d["language"] == "py"


def test_binary_content_is_reported_not_mangled(root: Path):
    """Returning mojibake would let the editor save it back and corrupt it."""
    (root / "b.bin").write_bytes(b"\xff\xfe\x00\x01")
    d = body(fs.read_text(str(root / "b.bin"), str(root)))
    assert d["binary"] is True
    assert d["text"] == ""


def test_a_large_file_is_truncated_and_says_so(root: Path, monkeypatch):
    monkeypatch.setattr(fs, "MAX_TEXT_BYTES", 10)
    (root / "big.txt").write_text("x" * 50)
    d = body(fs.read_text(str(root / "big.txt"), str(root)))
    assert d["truncated"] is True
    assert len(d["text"]) == 10
    assert d["byteSize"] == 50


def test_a_missing_file_is_404_not_500(root: Path):
    assert fs.read_text(str(root / "nope.txt"), str(root)).status_code == 404


def test_write_then_read_round_trips(root: Path):
    w = fs.write_text(str(root / "n.md"), "# hi", str(root))
    assert w.status_code == 200
    assert body(w)["byteSize"] == 4
    assert body(fs.read_text(str(root / "n.md"), str(root)))["text"] == "# hi"


def test_write_creates_missing_parents(root: Path):
    r = fs.write_text(str(root / "deep" / "x" / "n.md"), "hi", str(root))
    assert r.status_code == 200
    assert (root / "deep" / "x" / "n.md").read_text() == "hi"


def test_write_leaves_no_temp_file_behind(root: Path):
    fs.write_text(str(root / "n.md"), "hi", str(root))
    assert [p.name for p in root.iterdir()] == ["n.md"]


def test_write_replaces_atomically(root: Path):
    """A half-written file must never replace a whole one."""
    target = root / "n.md"
    target.write_text("original")
    fs.write_text(str(target), "replacement", str(root))
    assert target.read_text() == "replacement"


def test_list_returns_the_measured_shape(root: Path):
    (root / "dir").mkdir()
    (root / "f.txt").write_text("x")
    d = body(fs.list_directory(str(root), str(root)))
    assert set(d) == {"path", "parent", "entries", "root", "locked_root", "can_change_path"}
    names = [e["name"] for e in d["entries"]]
    assert names == ["dir", "f.txt"], "directories sort first, then case-insensitive"
    entry = next(e for e in d["entries"] if e["name"] == "f.txt")
    assert set(entry) == {"name", "path", "is_directory", "size", "mtime", "mime_type"}
    assert entry["is_directory"] is False


def test_the_root_reports_no_parent(root: Path):
    assert body(fs.list_directory(str(root), str(root)))["parent"] is None


def test_a_subdirectory_reports_its_parent(root: Path):
    (root / "sub").mkdir()
    assert body(fs.list_directory(str(root / "sub"), str(root)))["parent"] == str(root)


def test_listing_a_file_is_400(root: Path):
    (root / "f.txt").write_text("x")
    assert fs.list_directory(str(root / "f.txt"), str(root)).status_code == 400


def test_a_broken_symlink_does_not_break_the_listing(root: Path):
    (root / "good.txt").write_text("x")
    (root / "dangling").symlink_to(root / "gone.txt")
    d = body(fs.list_directory(str(root), str(root)))
    assert "good.txt" in [e["name"] for e in d["entries"]]


def test_mkdir_is_idempotent(root: Path):
    first = fs.make_directory(str(root / "d"), str(root))
    second = fs.make_directory(str(root / "d"), str(root))
    assert first.status_code == 200 and second.status_code == 200


def test_delivery_copies_the_bytes_into_the_sandbox(root: Path, tmp_path: Path):
    source = tmp_path / "upload.bin"
    source.write_bytes(b"payload")
    ok, detail = fs.deliver_attachment(str(root / "in" / "f.bin"), str(source), str(root))
    assert ok, detail
    assert (root / "in" / "f.bin").read_bytes() == b"payload"


def test_delivery_verifies_the_checksum(root: Path, tmp_path: Path):
    import hashlib

    source = tmp_path / "upload.bin"
    source.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    ok, detail = fs.deliver_attachment(
        str(root / "f.bin"), str(source), str(root), expected_checksum=digest
    )
    assert ok and "sha256 match" in detail


def test_delivery_fails_loudly_on_a_checksum_mismatch(root: Path, tmp_path: Path):
    source = tmp_path / "upload.bin"
    source.write_bytes(b"payload")
    ok, detail = fs.deliver_attachment(
        str(root / "f.bin"), str(source), str(root), expected_checksum="0" * 64
    )
    assert not ok
    assert "does not match the upload" in detail


def test_delivery_outside_the_root_is_refused(root: Path, tmp_path: Path):
    source = tmp_path / "upload.bin"
    source.write_bytes(b"x")
    target = root.parent / "escape.bin"
    ok, _ = fs.deliver_attachment(str(target), str(source), str(root))
    assert not ok
    assert not target.exists()


def test_delivery_reports_missing_source_bytes(root: Path, tmp_path: Path):
    ok, detail = fs.deliver_attachment(str(root / "f.bin"), str(tmp_path / "gone"), str(root))
    assert not ok
    assert "missing from the store" in detail


def test_delivery_leaves_no_temp_file(root: Path, tmp_path: Path):
    source = tmp_path / "upload.bin"
    source.write_bytes(b"x")
    fs.deliver_attachment(str(root / "f.bin"), str(source), str(root))
    assert [p.name for p in root.iterdir()] == ["f.bin"]
