"""Tests for WikipediaIngester (synthetic XML)."""

from __future__ import annotations

from pathlib import Path

from darwin.ingest.wikipedia import (
    WikipediaIngester,
    iter_articles,
    strip_mediawiki,
)


_SAMPLE_DUMP = """<?xml version="1.0"?>
<mediawiki>
  <page>
    <title>Neuron</title>
    <ns>0</ns>
    <revision>
      <text>A neuron is a [[cell]] in the [[nervous system]]. Neurons cause thoughts.</text>
    </revision>
  </page>
  <page>
    <title>Talk:Neuron</title>
    <ns>1</ns>
    <revision>
      <text>This is a talk page that should be skipped.</text>
    </revision>
  </page>
  <page>
    <title>Photon</title>
    <ns>0</ns>
    <revision>
      <text>A photon is a particle. Photons travel at light speed.</text>
    </revision>
  </page>
</mediawiki>
"""


def test_strip_mediawiki_wikilinks() -> None:
    text = strip_mediawiki("A neuron is a [[cell]] in the [[nervous system|nervous network]].")
    assert "[[" not in text
    assert "]]" not in text
    assert "cell" in text
    assert "nervous network" in text


def test_strip_mediawiki_templates_dropped() -> None:
    text = strip_mediawiki("Real text {{infobox|name=foo}} more text.")
    assert "{{" not in text
    assert "}}" not in text
    assert "Real text" in text


def test_strip_mediawiki_refs_dropped() -> None:
    text = strip_mediawiki("Real text <ref name='x'>citation</ref> more text.")
    assert "<ref" not in text
    assert "citation" not in text


def test_iter_articles_yields_mainspace_only(tmp_path: Path) -> None:
    path = tmp_path / "dump.xml"
    path.write_text(_SAMPLE_DUMP)
    titles = [a.title for a in iter_articles(path)]
    assert "Neuron" in titles
    assert "Photon" in titles
    assert "Talk:Neuron" not in titles


def test_wikipedia_ingester_processes_synthetic_dump(tmp_path: Path) -> None:
    path = tmp_path / "dump.xml"
    path.write_text(_SAMPLE_DUMP)
    ingester = WikipediaIngester()
    results = ingester.ingest_dump(path, max_articles=10)
    assert len(results) == 2
    assert ingester.articles_processed == 2
    assert ingester.facts_added > 0


def test_iter_articles_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_articles(tmp_path / "nope.xml")) == []
