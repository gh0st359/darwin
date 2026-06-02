"""WikipediaIngester — streams Wikipedia XML dumps one article at a time.

Uses ``xml.etree.ElementTree.iterparse`` so the full dump never lives in
memory. Each ``<page><revision><text>`` element becomes a candidate
article; MediaWiki markup is stripped to plain text via a small set of
deterministic rules (no markdown parser); the plain text is then
funnelled through ``DocumentIngester.ingest_text``.

Articles whose namespace is anything other than the default mainspace
(0) are skipped — that drops talk pages, category descriptions, user
pages, templates, modules, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET

from darwin.ingest.document import DocumentIngester, IngestResult


# MediaWiki markup → plain text rules. Each is conservative; we'd
# rather drop a malformed snippet than misparse it.
_LINK_RX = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_TEMPLATE_RX = re.compile(r"\{\{[^{}]*?\}\}")
_REF_RX = re.compile(r"<ref[^>]*?>.*?</ref>|<ref[^/]*?/>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RX = re.compile(r"<[^>]+>")
_BOLD_ITALIC_RX = re.compile(r"'{2,5}")
_HEADING_RX = re.compile(r"^={2,}\s*(.*?)\s*={2,}\s*$", re.MULTILINE)
_TABLE_RX = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_FILE_LINK_RX = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.IGNORECASE)


def strip_mediawiki(markup: str) -> str:
    """Strip MediaWiki markup → plain text. Conservative."""

    text = markup or ""
    # Iteratively unwrap nested templates: {{...}} until stable.
    for _ in range(5):
        new = _TEMPLATE_RX.sub("", text)
        if new == text:
            break
        text = new
    text = _TABLE_RX.sub("", text)
    text = _REF_RX.sub("", text)
    text = _FILE_LINK_RX.sub("", text)
    # Wikilinks: [[Target|surface]] → surface; [[Target]] → Target.
    text = _LINK_RX.sub(lambda m: m.group(2) or m.group(1), text)
    text = _HTML_TAG_RX.sub("", text)
    text = _BOLD_ITALIC_RX.sub("", text)
    text = _HEADING_RX.sub(r"\1.", text)
    # Bullets and numbered list markers at line start → drop the marker.
    text = re.sub(r"^[*#]+\s*", "", text, flags=re.MULTILINE)
    # Collapse whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class WikipediaArticle:
    title: str
    text: str
    namespace: int = 0


def iter_articles(dump_path: str | Path) -> Iterator[WikipediaArticle]:
    """Stream articles from a Wikipedia dump file. Skips non-mainspace."""

    target = Path(dump_path)
    if not target.exists():
        return
    # Namespaces use the XML tag prefix that varies between dump
    # versions; strip by local-name.
    ns_uri = ""
    try:
        for event, elem in ET.iterparse(str(target), events=("start", "end")):
            tag = elem.tag.split("}", 1)[-1] if "}" in elem.tag else elem.tag
            if event == "start" and ns_uri == "" and "}" in elem.tag:
                ns_uri = elem.tag.split("}", 1)[0].strip("{")
            if event == "end" and tag == "page":
                title = ""
                namespace = 0
                text = ""
                for child in list(elem):
                    ctag = child.tag.split("}", 1)[-1] if "}" in child.tag else child.tag
                    if ctag == "title":
                        title = child.text or ""
                    elif ctag == "ns":
                        try:
                            namespace = int((child.text or "0").strip())
                        except ValueError:
                            namespace = 0
                    elif ctag == "revision":
                        for grand in list(child):
                            gtag = grand.tag.split("}", 1)[-1] if "}" in grand.tag else grand.tag
                            if gtag == "text":
                                text = grand.text or ""
                                break
                # Free the page element's children.
                elem.clear()
                if namespace == 0 and title and text:
                    yield WikipediaArticle(title=title, text=text, namespace=namespace)
    except ET.ParseError:
        return


class WikipediaIngester:
    """Ingest a Wikipedia XML dump into the universe via DocumentIngester."""

    def __init__(self, document_ingester: DocumentIngester | None = None) -> None:
        self.document_ingester = document_ingester or DocumentIngester()
        self.articles_processed = 0
        self.facts_added = 0

    def ingest_dump(
        self,
        dump_path: str | Path,
        *,
        max_articles: int | None = None,
    ) -> list[IngestResult]:
        """Stream the dump. Returns per-article results."""

        results: list[IngestResult] = []
        for i, article in enumerate(iter_articles(dump_path)):
            if max_articles is not None and i >= max_articles:
                break
            plain = strip_mediawiki(article.text)
            res = self.document_ingester.ingest_text(
                plain, source=f"wikipedia:{article.title}",
            )
            results.append(res)
            self.articles_processed += 1
            self.facts_added += len(res.facts)
        return results


__all__ = [
    "WikipediaArticle",
    "WikipediaIngester",
    "iter_articles",
    "strip_mediawiki",
]
