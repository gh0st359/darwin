"""Per-user operator model: tracks conversational style across sessions.

Distinct from the v7 ``ObserverModeler`` which tracks operator *attention* and
*intervention probability*. This model is about how the user *converses* —
preferred reply length, topics they keep returning to, statements they have
made (agreements / disagreements with Darwin's prior claims). The reply
planner consults the model so Darwin can adapt its rhetorical shape to the
specific person on the other end of the socket.

Storage is intentionally additive: a single row keyed by ``user_id``. v5
chat callers that pass no user_id share a default "anonymous" model.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _StyleProfile:
    avg_utterance_words: float = 0.0
    samples: int = 0
    preferred_verbosity: str = "medium"  # "short" / "medium" / "long"

    def observe(self, word_count: int) -> None:
        self.samples += 1
        old_total = self.avg_utterance_words * (self.samples - 1)
        self.avg_utterance_words = (old_total + word_count) / self.samples
        if self.avg_utterance_words < 8:
            self.preferred_verbosity = "short"
        elif self.avg_utterance_words > 25:
            self.preferred_verbosity = "long"
        else:
            self.preferred_verbosity = "medium"


@dataclass
class OperatorModel:
    user_id: str = "anonymous"
    style_profile: _StyleProfile = field(default_factory=_StyleProfile)
    interests: Counter = field(default_factory=Counter)
    agreements: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    last_seen_at: float = 0.0
    first_seen_at: float = field(default_factory=time.time)

    def observe(self, turn: str) -> None:
        """Update the model from a single user turn."""

        words = turn.split()
        self.style_profile.observe(len(words))
        for word in words:
            token = word.strip(".,!?:;\"'()[]{}").lower()
            if len(token) > 4 and not token.startswith("/"):
                self.interests[token] += 1
        # Coarse stance detection.
        lowered = turn.lower()
        agreement_markers = ("yes", "agreed", "right", "exactly", "correct", "true")
        disagreement_markers = ("no", "wrong", "incorrect", "disagree", "false", "actually")
        if any(marker in lowered for marker in agreement_markers):
            self.agreements.append(turn[:120])
            self.agreements = self.agreements[-32:]
        if any(marker in lowered for marker in disagreement_markers):
            self.disagreements.append(turn[:120])
            self.disagreements = self.disagreements[-32:]
        self.last_seen_at = time.time()

    def preferred_length(self, plan_mode: str | None = None) -> str:
        """Translate verbosity to the discourse planner's target_length."""

        return self.style_profile.preferred_verbosity

    def top_interests(self, limit: int = 8) -> list[str]:
        return [word for word, _ in self.interests.most_common(limit)]

    def to_record(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "samples": self.style_profile.samples,
            "avg_words": round(self.style_profile.avg_utterance_words, 2),
            "preferred_verbosity": self.style_profile.preferred_verbosity,
            "top_interests": self.top_interests(),
            "agreements": list(self.agreements),
            "disagreements": list(self.disagreements),
            "last_seen_at": self.last_seen_at,
            "first_seen_at": self.first_seen_at,
        }


class OperatorModelRegistry:
    """Per-user_id registry. Default anonymous model is always available."""

    def __init__(self) -> None:
        self._models: dict[str, OperatorModel] = {}

    def get(self, user_id: str | None = None) -> OperatorModel:
        key = user_id or "anonymous"
        model = self._models.get(key)
        if model is None:
            model = OperatorModel(user_id=key)
            self._models[key] = model
        return model

    def known_users(self) -> list[str]:
        return list(self._models)

    def __len__(self) -> int:
        return len(self._models)
