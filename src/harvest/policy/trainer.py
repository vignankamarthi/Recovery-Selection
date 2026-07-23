"""ACT baseline trainer interface (Part 1, step 1.11).

The imitation baseline is ACT (Action Chunking with Transformers, the LeRobot implementation). Its
torch/LeRobot dependency is heavy and GPU-bound, so it sits BEHIND a small interface: the pipeline
(dataset -> fit -> evaluate -> per-condition metrics) is defined and tested here without torch, and
the real trainer plugs in on AICR.

  - `Trainer` (Protocol): `fit(train)` then `evaluate(eval_set) -> TrainMetrics`.
  - `StubTrainer`: a torch-free stand-in that proves the machinery locally. It is the condition
    majority-class baseline (the audit's static-appearance shortcut reference), so its numbers are a
    meaningful FLOOR the real ACT must beat, not noise. The local smoke test uses it.
  - `LeRobotACTTrainer`: the real trainer. Same interface, lazily imports LeRobot, trains ACT on the
    materialized stream dataset, and rolls the policy out in sim on val/test for the real
    label-exposure metric. Runs on AICR (human-gated), never in the local test path.

Backend-agnostic and MuJoCo-free, like the control split. Reports per-condition label-exposure
success, Padir's criterion (>=75% nominal, >=50% damaged on the physical data).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from schema.episode import Outcome, RecordedEpisode


@dataclass
class TrainMetrics:
    """Eval outcome of a trained policy. `per_condition_success` maps a condition to the policy's
    label-exposure success rate on that condition's eval episodes (a rollout success rate for the
    real ACT; the majority-baseline accuracy for the stub). `overall_success` is the pooled rate."""

    per_condition_success: dict[str, float] = field(default_factory=dict)
    overall_success: float = 0.0
    n_train: int = 0
    n_eval: int = 0


@runtime_checkable
class Trainer(Protocol):
    """Trains an imitation policy on demonstration episodes and evaluates label-exposure success."""

    def fit(self, train: Sequence[RecordedEpisode]) -> "Trainer": ...

    def evaluate(self, eval_set: Sequence[RecordedEpisode]) -> TrainMetrics: ...


def _succeeded(rec: RecordedEpisode) -> bool:
    return rec.episode.outcome is Outcome.SUCCESS


class StubTrainer:
    """Torch-free condition majority-class baseline. `fit` learns each condition's majority outcome
    from train; `evaluate` predicts that per eval episode and scores accuracy per condition. This is
    the static-appearance shortcut reference (it reads no streams), so it proves the train/eval
    machinery AND gives a meaningful floor the real ACT must beat."""

    def __init__(self) -> None:
        self._pred: dict[str, bool] = {}
        self._n_train = 0

    def fit(self, train: Sequence[RecordedEpisode]) -> "StubTrainer":
        by_cond: dict[str, Counter] = defaultdict(Counter)
        for rec in train:
            by_cond[rec.episode.condition.value][_succeeded(rec)] += 1
        self._pred = {c: cc[True] >= cc[False] for c, cc in by_cond.items()}
        self._n_train = len(train)
        return self

    def evaluate(self, eval_set: Sequence[RecordedEpisode]) -> TrainMetrics:
        hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # [correct, total] per condition
        overall = [0, 0]
        for rec in eval_set:
            cond = rec.episode.condition.value
            pred = self._pred.get(cond, True)                      # unseen condition -> optimistic
            correct = int(pred == _succeeded(rec))
            hits[cond][0] += correct
            hits[cond][1] += 1
            overall[0] += correct
            overall[1] += 1
        per_cond = {c: (h[0] / h[1] if h[1] else 0.0) for c, h in hits.items()}
        return TrainMetrics(
            per_condition_success=per_cond,
            overall_success=(overall[0] / overall[1] if overall[1] else 0.0),
            n_train=self._n_train,
            n_eval=len(eval_set),
        )


class LeRobotACTTrainer:
    """The real ACT baseline (LeRobot). Same `Trainer` interface, runs on AICR only. LeRobot/torch
    are imported lazily so importing this module stays torch-free. Left as an interface stub until
    the human-gated cluster run: materialize the stream dataset on the cluster, build (observation,
    action) pairs from the recorded streams (the demonstration joint trajectory is the action
    target), train ACT on the train split, then roll the policy out in sim on val/test for
    label-exposure success. Do not call in the local test path."""

    def __init__(self, chunk_size: int = 100, n_steps: int = 100_000) -> None:
        self.chunk_size = chunk_size
        self.n_steps = n_steps

    def fit(self, train: Sequence[RecordedEpisode]) -> "LeRobotACTTrainer":
        raise NotImplementedError(
            "LeRobotACTTrainer runs on AICR (torch + GPU). Materialize the stream dataset on the "
            "cluster, then train there via the human-gated flow. Use StubTrainer for local tests."
        )

    def evaluate(self, eval_set: Sequence[RecordedEpisode]) -> TrainMetrics:
        raise NotImplementedError("LeRobotACTTrainer eval rolls the trained policy out in sim on AICR.")
