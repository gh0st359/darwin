"""V-Ingest — pure-Python knowledge ingestion at scale.

Four sources of bulk knowledge feed into Darwin's universe via the
``IngestPipeline``:

  * :class:`DocumentIngester` — text and HTML.
  * :class:`WikipediaIngester` — XML dumps streamed one article at a time.
  * :class:`ArxivIngester` — LaTeX abstracts + HTML mirrors.
  * :class:`CodeRepoIngester` — git-walked symbol table.

A hand-rolled :class:`NLParser` (tokenize → POS tag → relation
extraction) produces ``(subject, predicate, object)`` Facts. The
pipeline fuses Facts into the concept graph, activates the
corresponding mesh cells, and emits bus events with rolling throughput.

No LLM. No pretrained weights. Pure Python.
"""

from darwin.ingest.arxiv import ArxivIngester, ArxivPaper, strip_latex
from darwin.ingest.code_repo import CodeRepoIngester, RepoIngestResult, Symbol
from darwin.ingest.document import DocumentIngester, IngestResult
from darwin.ingest.nl_parser import (
    Fact,
    NLParser,
    Token,
    extract_facts,
    named_entities,
    pos_tag,
    sentences,
    tokenize,
)
from darwin.ingest.pipeline import IngestPipeline, IngestStats
from darwin.ingest.wikipedia import (
    WikipediaArticle,
    WikipediaIngester,
    iter_articles,
    strip_mediawiki,
)


__all__ = [
    "ArxivIngester",
    "ArxivPaper",
    "CodeRepoIngester",
    "DocumentIngester",
    "Fact",
    "IngestPipeline",
    "IngestResult",
    "IngestStats",
    "NLParser",
    "RepoIngestResult",
    "Symbol",
    "Token",
    "WikipediaArticle",
    "WikipediaIngester",
    "extract_facts",
    "iter_articles",
    "named_entities",
    "pos_tag",
    "sentences",
    "strip_latex",
    "strip_mediawiki",
    "tokenize",
]
