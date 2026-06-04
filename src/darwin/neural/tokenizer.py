"""NeuralTokenizer — deterministic, BPE-flavoured tokenizer over Darwin's corpus.

The tokenizer is *operator-grown*: the only vocabulary that exists is the
vocabulary the operator has actually fed Darwin. No pretrained merges, no
seeded vocab from a model. Two stages:

  1. **Word split.** Lowercase-and-strip-punctuation pass to produce a
     reproducible word stream from arbitrary text. Identical to what
     ``darwin.ingest.nl_parser.tokenize`` does at the POS layer, so the
     two pipelines stay aligned.

  2. **BPE-like merge.** Optional merge table mapping ``(a, b) -> ab``
     applied left-to-right. New merges can be learned online from corpus
     statistics; the merge table is the *vocabulary*. Merges are
     persisted to disk so a restart picks up where training stopped.

The merge process is not the focus of V-Neural — the embedding learner
is. The tokenizer's role is to give the embedding learner stable,
reproducible token ids. The merge table grows slowly compared to the
embedding state, which is the part that compounds.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_WORD_RE = re.compile(r"[a-z0-9]+(?:[_'-][a-z0-9]+)*")


def split_words(text: str) -> list[str]:
    """Reproducible whitespace+punctuation split → lowercase word list."""

    return _WORD_RE.findall(text.lower())


@dataclass
class NeuralTokenizer:
    """Word-level tokenizer with optional learned BPE-like merges."""

    merges: list[tuple[str, str]] = field(default_factory=list)
    merge_index: dict[tuple[str, str], int] = field(default_factory=dict)
    token_counts: Counter = field(default_factory=Counter)
    _total_chunks: int = 0

    def __post_init__(self) -> None:
        if self.merges and not self.merge_index:
            self.merge_index = {pair: i for i, pair in enumerate(self.merges)}

    # -- core API ----------------------------------------------------------- #

    def tokenize(self, text: str) -> list[str]:
        """Return the token stream for ``text``."""

        words = split_words(text)
        if not self.merges:
            self._observe(words)
            return words
        merged = self._apply_merges(words)
        self._observe(merged)
        return merged

    def _apply_merges(self, words: list[str]) -> list[str]:
        # Greedy left-to-right merge: scan pairs, apply the lowest-index
        # merge available, restart the scan after each merge.
        out = list(words)
        i = 0
        while i < len(out) - 1:
            pair = (out[i], out[i + 1])
            if pair in self.merge_index:
                out[i] = out[i] + out[i + 1]
                del out[i + 1]
                if i > 0:
                    i -= 1
                continue
            i += 1
        return out

    def _observe(self, tokens: list[str]) -> None:
        self.token_counts.update(tokens)
        self._total_chunks += 1

    # -- vocab growth ------------------------------------------------------- #

    def learn_merges(self, corpus: Iterable[str], *, max_merges: int = 1) -> int:
        """Mine pair frequencies across ``corpus``; add up to ``max_merges`` new merges.

        This is intentionally minimal — vocabulary growth is a slow background
        activity, not a per-chunk operation. The bulk of the learned signal
        lives in embeddings, not in merge selection.
        """

        pair_counts: Counter[tuple[str, str]] = Counter()
        for text in corpus:
            tokens = split_words(text)
            tokens = self._apply_merges(tokens) if self.merges else tokens
            for a, b in zip(tokens, tokens[1:]):
                pair_counts[(a, b)] += 1
        added = 0
        for pair, _count in pair_counts.most_common():
            if pair in self.merge_index:
                continue
            self.merges.append(pair)
            self.merge_index[pair] = len(self.merges) - 1
            added += 1
            if added >= max_merges:
                break
        return added

    # -- persistence -------------------------------------------------------- #

    def to_record(self) -> dict:
        return {
            "merges": [list(p) for p in self.merges],
            "token_counts": dict(self.token_counts),
            "total_chunks": self._total_chunks,
        }

    @classmethod
    def from_record(cls, record: dict) -> "NeuralTokenizer":
        merges = [tuple(p) for p in record.get("merges", [])]
        return cls(
            merges=merges,
            merge_index={pair: i for i, pair in enumerate(merges)},
            token_counts=Counter(record.get("token_counts", {})),
            _total_chunks=int(record.get("total_chunks", 0)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_record()))

    @classmethod
    def load(cls, path: str | Path) -> "NeuralTokenizer":
        return cls.from_record(json.loads(Path(path).read_text()))

    # -- introspection ------------------------------------------------------ #

    def vocab_size(self) -> int:
        return len(self.token_counts)

    def stats(self) -> dict:
        return {
            "vocab_size": self.vocab_size(),
            "merge_count": len(self.merges),
            "chunks_observed": self._total_chunks,
        }


__all__ = ["NeuralTokenizer", "split_words"]
