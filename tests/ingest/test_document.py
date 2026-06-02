"""Tests for DocumentIngester."""

from __future__ import annotations

from pathlib import Path

from darwin.ingest.document import DocumentIngester


def test_ingest_text_extracts_facts() -> None:
    ingester = DocumentIngester()
    result = ingester.ingest_text(
        "A neuron is a cell. Rain causes flooding."
    )
    assert len(result.facts) == 2
    assert result.error == ""


def test_ingest_text_empty_yields_zero_facts() -> None:
    ingester = DocumentIngester()
    result = ingester.ingest_text("")
    assert result.facts == []


def test_ingest_html_strips_tags_and_extracts_facts() -> None:
    ingester = DocumentIngester()
    html = """<html><body>
        <p>A photon is a particle.</p>
        <script>console.log('skip me');</script>
        <p>Rain causes flooding.</p>
    </body></html>"""
    result = ingester.ingest_html(html)
    predicates = {f.predicate for f in result.facts}
    assert "is_a" in predicates
    assert "causes" in predicates


def test_ingest_html_drops_script_and_style() -> None:
    ingester = DocumentIngester()
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body>A widget is a gadget.</body></html>"
    )
    result = ingester.ingest_html(html)
    # The 'color:red' should never become a parsed sentence.
    for f in result.facts:
        assert "color" not in f.source_sentence.lower()
    assert any(f.subject == "widget" for f in result.facts)


def test_ingest_file_text(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("A maple is a tree.")
    ingester = DocumentIngester()
    result = ingester.ingest_file(f)
    assert result.error == ""
    assert any(fa.subject == "maple" for fa in result.facts)


def test_ingest_file_html(tmp_path: Path) -> None:
    f = tmp_path / "note.html"
    f.write_text("<html><body>A photon is a particle.</body></html>")
    ingester = DocumentIngester()
    result = ingester.ingest_file(f)
    assert result.error == ""
    assert any(fa.predicate == "is_a" for fa in result.facts)


def test_ingest_file_missing(tmp_path: Path) -> None:
    ingester = DocumentIngester()
    result = ingester.ingest_file(tmp_path / "nope.txt")
    assert result.error
    assert "not found" in result.error.lower()
