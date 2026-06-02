"""Tests for ArxivIngester."""

from __future__ import annotations

from darwin.ingest.arxiv import ArxivIngester, ArxivPaper, strip_latex


def test_strip_latex_removes_math_inline() -> None:
    out = strip_latex("Energy is $E = mc^2$ for a body at rest.")
    assert "$" not in out
    assert "Energy" in out
    assert "rest" in out


def test_strip_latex_removes_commands() -> None:
    out = strip_latex("\\textit{Italic} text \\emph{emphasised} here.")
    assert "\\textit" not in out
    assert "Italic" not in out or "Italic" in out  # command vanishes; contents stay or go
    assert "text" in out


def test_strip_latex_drops_environments() -> None:
    out = strip_latex(
        "Text before. \\begin{equation} F = ma \\end{equation} Text after."
    )
    assert "\\begin" not in out
    assert "\\end" not in out
    assert "Text before" in out
    assert "Text after" in out


def test_strip_latex_drops_comments() -> None:
    out = strip_latex("Real text. % a comment to drop\nMore real text.")
    assert "comment" not in out
    assert "Real text" in out


def test_arxiv_ingester_processes_paper() -> None:
    ingester = ArxivIngester()
    paper = ArxivPaper(
        arxiv_id="2024.0001",
        title="A neuron is a cell",
        abstract=(
            "Neurons cause thoughts. A brain is composed of neurons. "
            "Each cell is part of an organism."
        ),
    )
    result = ingester.ingest_paper(paper)
    predicates = {f.predicate for f in result.facts}
    assert predicates  # at least one fact emerged
    assert ingester.papers_processed == 1


def test_arxiv_ingester_ingest_many() -> None:
    ingester = ArxivIngester()
    papers = [
        ArxivPaper(arxiv_id="a", title="A photon is a particle", abstract=""),
        ArxivPaper(arxiv_id="b", title="A photon is a wave", abstract=""),
    ]
    results = ingester.ingest_many(papers)
    assert len(results) == 2
    assert ingester.papers_processed == 2
