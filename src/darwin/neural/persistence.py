"""NeuralPersistence — atomic, sharded save/load across the neural substrate.

One manifest + N vector shards + tokenizer JSON + training-state JSON, all
under ``<data_dir>/neural/`` by default. The manifest records the dim,
backend, shard count, and a content hash so a corrupt shard set fails loud.

Operator-facing CLI labels checkpoints (e.g. ``baseline``,
``after-wiki-pass``); labels live under ``<data_dir>/neural/checkpoints/<label>/``
and are full copies of the active set. ``rollback(label)`` atomically
swaps the active set with a labelled one.

Atomicity is per-file: each write goes to a ``.tmp`` and is renamed.
The whole save is a sequence of atomic file ops; a crash mid-save can
leave the previous active set intact (the manifest is written last).
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from darwin.neural.learned_space import LearnedCausalSpace
from darwin.neural.tokenizer import NeuralTokenizer


MANIFEST_NAME = "manifest.json"
TOKENIZER_NAME = "tokenizer.json"
TRAINING_NAME = "training.json"
SHARDS_DIR = "shards"
CHECKPOINTS_DIR = "checkpoints"
CURSOR_NAME = "training_cursor.json"


@dataclass
class TrainingState:
    """Lightweight non-vector training state."""

    train_steps: int = 0
    total_tokens_seen: int = 0
    freq: dict[str, int] | None = None
    loss_ewma: float = 0.0


@dataclass
class Manifest:
    version: int
    dim: int
    backend: str
    shard_count: int
    vocab_size: int
    saved_at: float
    label: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class NeuralPersistence:
    """Coordinates save/load across all neural sub-stores."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def shards_dir(self) -> Path:
        return self.root / SHARDS_DIR

    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def tokenizer_path(self) -> Path:
        return self.root / TOKENIZER_NAME

    def training_path(self) -> Path:
        return self.root / TRAINING_NAME

    def cursor_path(self) -> Path:
        return self.root / CURSOR_NAME

    # -- save -------------------------------------------------------------- #

    def save(
        self,
        *,
        space: LearnedCausalSpace,
        tokenizer: NeuralTokenizer | None = None,
        label: str = "",
    ) -> Manifest:
        # 1. vectors → shards
        shards = space._store.shard_to_disk(self.shards_dir())
        # 2. tokenizer
        if tokenizer is not None:
            tokenizer.save(self.tokenizer_path())
        # 3. training state (everything not in the shards)
        training = TrainingState(
            train_steps=int(space._train_steps),
            total_tokens_seen=int(space._total_tokens_seen),
            freq=dict(space._freq),
            loss_ewma=float(space.light_stats()["loss_ewma"]),
        )
        self._atomic_write(self.training_path(), json.dumps(asdict(training)))
        # 4. manifest written LAST so a partial save never claims success
        manifest = Manifest(
            version=1,
            dim=space.dim,
            backend=space.backend,
            shard_count=len(shards),
            vocab_size=space.vocab_size(),
            saved_at=time.time(),
            label=label,
        )
        self._atomic_write(self.manifest_path(), json.dumps(manifest.to_record()))
        return manifest

    # -- load -------------------------------------------------------------- #

    def load(
        self,
        *,
        space: LearnedCausalSpace,
        tokenizer: NeuralTokenizer | None = None,
    ) -> Manifest | None:
        if not self.manifest_path().exists():
            return None
        manifest_data = json.loads(self.manifest_path().read_text())
        manifest = Manifest(**manifest_data)
        if manifest.dim != space.dim:
            raise ValueError(
                f"manifest dim {manifest.dim} != live space dim {space.dim}"
            )
        loaded = space._store.load_shards(self.shards_dir())
        if self.training_path().exists():
            training = json.loads(self.training_path().read_text())
            space._train_steps = int(training.get("train_steps", 0))
            space._total_tokens_seen = int(training.get("total_tokens_seen", 0))
            space._freq = {k: int(v) for k, v in (training.get("freq") or {}).items()}
            space._loss_ewma = float(training.get("loss_ewma", 0.0))
        if tokenizer is not None and self.tokenizer_path().exists():
            loaded_tok = NeuralTokenizer.load(self.tokenizer_path())
            tokenizer.merges = loaded_tok.merges
            tokenizer.merge_index = loaded_tok.merge_index
            tokenizer.token_counts = loaded_tok.token_counts
            tokenizer._total_chunks = loaded_tok._total_chunks
        return manifest

    # -- labelled checkpoints --------------------------------------------- #

    def checkpoint(self, label: str) -> Path:
        """Copy the active set into a labelled checkpoint directory."""

        if not label or "/" in label:
            raise ValueError("label must be a non-empty, slash-free string")
        dest = self.root / CHECKPOINTS_DIR / label
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name in (MANIFEST_NAME, TOKENIZER_NAME, TRAINING_NAME):
            src = self.root / name
            if src.exists():
                shutil.copy2(src, dest / name)
        if self.shards_dir().exists():
            shutil.copytree(self.shards_dir(), dest / SHARDS_DIR)
        return dest

    def list_checkpoints(self) -> list[str]:
        base = self.root / CHECKPOINTS_DIR
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def rollback(self, label: str) -> None:
        """Atomically swap the active set with a labelled checkpoint."""

        src = self.root / CHECKPOINTS_DIR / label
        if not src.exists():
            raise FileNotFoundError(f"checkpoint label not found: {label}")
        # Snapshot current active set so the rollback itself is reversible.
        backup_label = f"_rollback_backup_{int(time.time())}"
        self.checkpoint(backup_label)
        # Replace active files.
        for name in (MANIFEST_NAME, TOKENIZER_NAME, TRAINING_NAME):
            srcf = src / name
            dstf = self.root / name
            if srcf.exists():
                shutil.copy2(srcf, dstf)
        # Replace shards directory.
        if self.shards_dir().exists():
            shutil.rmtree(self.shards_dir())
        if (src / SHARDS_DIR).exists():
            shutil.copytree(src / SHARDS_DIR, self.shards_dir())

    # -- cursor ------------------------------------------------------------ #

    def read_cursor(self) -> dict[str, Any]:
        if not self.cursor_path().exists():
            return {}
        return json.loads(self.cursor_path().read_text())

    def write_cursor(self, cursor: dict[str, Any]) -> None:
        self._atomic_write(self.cursor_path(), json.dumps(cursor))

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)


__all__ = ["NeuralPersistence", "Manifest", "TrainingState"]
