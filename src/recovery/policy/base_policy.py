"""Frozen base-policy loader (Part 2, step 2.1).

The recovery study rides on the ACT baseline trained in Part 1 (step 1.11) and treats it as a FROZEN
black box: inference or LoRA only, never full training (the recovery layer trains only the selector).
Two signals for the competence model are read off this frozen policy -- latent-state density and
small-ensemble action disagreement -- so the base policy must expose a latent embedding and a small
ensemble of action heads, not just a point prediction.

Mirrors the Part-1 trainer pattern (`harvest/policy/trainer.py`): a `BasePolicy` Protocol, a torch-free
`StubBasePolicy` that runs locally for TDD and the recovery smoke test, and a `FrozenACTPolicy` that
lazily imports torch/LeRobot and runs on AICR only.

FENCE: imports `schema`-nothing here (stdlib only), never `harvest`. The stub reuses no Part-1 code;
the real ACT is loaded from a checkpoint path, not from `harvest`.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable

Observation = object   # opaque at this layer (a dict of modality arrays, a tensor ref, ...)
Action = Sequence[float]


@runtime_checkable
class BasePolicy(Protocol):
    """A frozen imitation policy the recovery layer treats as a black box.

    `frozen` is always True by contract (recovery never trains the backbone). Beyond a point
    `predict`, it exposes a `latent` embedding and an `action_ensemble` so the competence model can
    read latent-state density and ensemble disagreement.
    """

    @property
    def frozen(self) -> bool: ...

    def predict(self, observation: Observation) -> Action: ...

    def latent(self, observation: Observation) -> list[float]: ...

    def action_ensemble(self, observation: Observation) -> list[list[float]]: ...


def _hash_floats(key: str, n: int) -> list[float]:
    """Deterministic pseudo-random floats in [0, 1) from a string key (stdlib-only, no numpy)."""
    out: list[float] = []
    i = 0
    while len(out) < n:
        h = hashlib.sha256(f"{key}:{i}".encode()).digest()
        for b in h:
            out.append(b / 255.0)
            if len(out) >= n:
                break
        i += 1
    return out


class StubBasePolicy:
    """Torch-free deterministic stand-in for the frozen ACT baseline.

    It is NOT a trained policy; it exists so the recovery machinery (competence signals, the arms, the
    grid, the selector) runs and is testable locally without torch or a checkpoint. Its `predict`,
    `latent`, and `action_ensemble` are deterministic functions of the observation, so it behaves like
    a frozen policy (same obs -> same output), which is exactly the property the recovery loop needs.
    """

    def __init__(self, action_dim: int = 7, latent_dim: int = 32, ensemble_size: int = 5) -> None:
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.ensemble_size = ensemble_size

    @property
    def frozen(self) -> bool:
        return True

    @staticmethod
    def _key(observation: Observation) -> str:
        try:
            return repr(sorted(observation.items()))  # type: ignore[attr-defined]
        except AttributeError:
            return repr(observation)

    def predict(self, observation: Observation) -> Action:
        return _hash_floats(self._key(observation) + "|act", self.action_dim)

    def latent(self, observation: Observation) -> list[float]:
        return _hash_floats(self._key(observation) + "|z", self.latent_dim)

    def action_ensemble(self, observation: Observation) -> list[list[float]]:
        return [
            _hash_floats(f"{self._key(observation)}|ens{m}", self.action_dim)
            for m in range(self.ensemble_size)
        ]


class FrozenACTPolicy:
    """The real frozen ACT baseline (LeRobot), loaded from a checkpoint. Same `BasePolicy` interface,
    runs on AICR only. torch/LeRobot are imported lazily inside the methods so importing this module
    stays torch-free (the fence's torch-quarantine). Left as an interface stub until the human-gated
    cluster run wires in the checkpoint + latent/ensemble reads. Do not call in the local test path."""

    def __init__(self, checkpoint: str, ensemble_size: int = 5) -> None:
        self.checkpoint = checkpoint
        self.ensemble_size = ensemble_size
        self._policy = None  # lazily loaded on the cluster

    @property
    def frozen(self) -> bool:
        return True

    def _load(self):
        raise NotImplementedError(
            "FrozenACTPolicy loads a LeRobot ACT checkpoint with torch on AICR (GPU). Wire this in "
            "during the human-gated cluster run; use StubBasePolicy for local tests."
        )

    def predict(self, observation: Observation) -> Action:
        self._load()
        raise NotImplementedError

    def latent(self, observation: Observation) -> list[float]:
        self._load()
        raise NotImplementedError

    def action_ensemble(self, observation: Observation) -> list[list[float]]:
        self._load()
        raise NotImplementedError
