"""CLI: darwin train {ingest|stream|status|checkpoint|probe|rollback}."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from darwin.cli import main as cli_main


def _run(argv: list[str], capsys, monkeypatch=None, stdin: str = ""):
    if stdin and monkeypatch is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    rc = cli_main(argv)
    captured = capsys.readouterr()
    return rc, captured.out


def test_train_status_smoke(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    rc, out = _run(["train", "status"], capsys)
    assert rc == 0
    data = json.loads(out)
    assert "vocab_size" in data


def test_train_ingest_file_grows_vocab(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "alpha beta gamma delta\n"
        "alpha gamma epsilon zeta\n"
        "beta gamma theta iota\n",
        encoding="utf-8",
    )
    rc, out = _run(["train", "ingest", "--path", str(corpus)], capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["files_ingested_this_run"] == 1
    assert data["vocab_size"] >= 5


def test_train_ingest_resumes_skipping_known_files(
    tmp_path: Path, capsys, monkeypatch,
):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (corpus_dir / "b.txt").write_text("delta epsilon zeta", encoding="utf-8")

    rc1, _ = _run(["train", "ingest", "--path", str(corpus_dir)], capsys)
    assert rc1 == 0

    rc2, out2 = _run(["train", "ingest", "--path", str(corpus_dir)], capsys)
    assert rc2 == 0
    data = json.loads(out2)
    # Second run sees no new files because the cursor remembered them.
    assert data["files_ingested_this_run"] == 0


def test_train_stream_from_stdin(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    rc, out = _run(
        ["train", "stream"], capsys, monkeypatch=monkeypatch,
        stdin="alpha beta gamma\nbeta gamma delta\n",
    )
    assert rc == 0
    data = json.loads(out)
    assert data["chunks"] == 2
    assert data["vocab_size"] >= 4


def test_train_probe_returns_diagnostic_payload(
    tmp_path: Path, capsys, monkeypatch,
):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma alpha beta gamma", encoding="utf-8")
    _run(["train", "ingest", "--path", str(corpus)], capsys)
    rc, out = _run(["train", "probe", "alpha"], capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["token"] == "alpha"
    assert isinstance(data["nearest"], list)


def test_train_checkpoint_and_list(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("foo bar baz", encoding="utf-8")
    _run(["train", "ingest", "--path", str(corpus)], capsys)
    rc, out = _run(["train", "checkpoint", "--label", "v1"], capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["label"] == "v1"
    rc, out = _run(["train", "list-checkpoints"], capsys)
    assert rc == 0
    assert "v1" in out


def test_train_rollback_restores_prior_state(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    corpus_a = tmp_path / "a.txt"
    corpus_a.write_text("alpha beta gamma", encoding="utf-8")
    corpus_b = tmp_path / "b.txt"
    corpus_b.write_text("delta epsilon zeta theta iota", encoding="utf-8")

    _run(["train", "ingest", "--path", str(corpus_a)], capsys)
    _run(["train", "checkpoint", "--label", "baseline"], capsys)
    rc, out_pre = _run(["train", "status"], capsys)
    pre = json.loads(out_pre)
    pre_vocab = pre["vocab_size"]

    _run(["train", "ingest", "--path", str(corpus_b)], capsys)
    rc, out_post = _run(["train", "status"], capsys)
    post = json.loads(out_post)
    assert post["vocab_size"] > pre_vocab

    rc, _ = _run(["train", "rollback", "baseline"], capsys)
    assert rc == 0
    rc, out_after = _run(["train", "status"], capsys)
    after = json.loads(out_after)
    assert after["vocab_size"] == pre_vocab


def test_train_ingest_missing_path_returns_error(
    tmp_path: Path, capsys, monkeypatch,
):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    rc, _ = _run(["train", "ingest", "--path", str(tmp_path / "nope")], capsys)
    assert rc == 2


def test_train_invalid_subcommand_returns_error(
    tmp_path: Path, capsys, monkeypatch,
):
    monkeypatch.setenv("DARWIN_DATA_DIR", str(tmp_path))
    # argparse handles "darwin train" alone — without a subcommand. We
    # exercise the "no train_command" path by faking attribute access.
    rc, out = _run(["train"], capsys)
    # Either parser rejects (rc != 0) or our handler prints usage and returns 1.
    assert rc == 1
    assert "darwin train" in out.lower()
