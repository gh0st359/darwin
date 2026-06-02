"""ArxivIngester — LaTeX abstracts + HTML mirrors.

The full LaTeX of an arXiv paper is too noisy for reliable fact
extraction. The strategy is: use the abstract (which is usually clean
LaTeX or plain English) as the main input, plus the title; if a HTML
mirror is supplied (arxiv-vanity / ar5iv), fall back to the document
ingester's HTML path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from darwin.ingest.document import DocumentIngester, IngestResult


# LaTeX → plain text rules. Conservative.
_LATEX_COMMAND_RX = re.compile(r"\\[a-zA-Z]+\*?(?:\{[^}]*\}|\[[^\]]*\])*")
_LATEX_ENV_RX = re.compile(r"\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\}", re.DOTALL)
_LATEX_MATH_INLINE_RX = re.compile(r"\$[^$]+\$")
_LATEX_MATH_DISPLAY_RX = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_LATEX_COMMENT_RX = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def strip_latex(text: str) -> str:
    """Strip LaTeX markup → plain text. Conservative."""

    if not text:
        return ""
    text = _LATEX_COMMENT_RX.sub("", text)
    text = _LATEX_ENV_RX.sub("", text)
    text = _LATEX_MATH_DISPLAY_RX.sub(" ", text)
    text = _LATEX_MATH_INLINE_RX.sub(" ", text)
    text = _LATEX_COMMAND_RX.sub("", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    html_mirror: str = ""


class ArxivIngester:
    """Ingest arXiv abstracts (and optionally HTML mirrors) into facts."""

    def __init__(self, document_ingester: DocumentIngester | None = None) -> None:
        self.document_ingester = document_ingester or DocumentIngester()
        self.papers_processed = 0
        self.facts_added = 0

    def ingest_paper(self, paper: ArxivPaper) -> IngestResult:
        # Build a clean preamble from the title + stripped abstract.
        title_clean = strip_latex(paper.title)
        abstract_clean = strip_latex(paper.abstract)
        combined = f"{title_clean}. {abstract_clean}".strip()
        result = self.document_ingester.ingest_text(
            combined, source=f"arxiv:{paper.arxiv_id}",
        )
        if paper.html_mirror:
            html_result = self.document_ingester.ingest_html(
                paper.html_mirror, source=f"arxiv-html:{paper.arxiv_id}",
            )
            # Merge facts; keep both source sentences.
            result.facts.extend(html_result.facts)
        self.papers_processed += 1
        self.facts_added += len(result.facts)
        return result

    def ingest_many(self, papers: Iterable[ArxivPaper]) -> list[IngestResult]:
        return [self.ingest_paper(p) for p in papers]


__all__ = ["ArxivIngester", "ArxivPaper", "strip_latex"]
