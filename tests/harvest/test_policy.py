"""Tests for the ACT baseline trainer interface (1.11). Torch-free: exercises the train/eval
machinery with a StubTrainer on synthetic episodes, so no torch and no sim. The real LeRobot ACT
trainer implements the same interface and runs on AICR."""

from harvest.policy.trainer import StubTrainer, Trainer, TrainMetrics
from schema.episode import (
    ConditionClass,
    Episode,
    Label,
    LabelProvenance,
    Outcome,
    RecordedEpisode,
)


def _ep(eid: str, cond: ConditionClass, success: bool) -> RecordedEpisode:
    outcome = Outcome.SUCCESS if success else Outcome.FAILURE
    ep = Episode(eid, f"{cond.value}-{eid}", cond, outcome=outcome,
                 labels=[Label("label_visible", success, LabelProvenance.AUTO_VISION)])
    return RecordedEpisode(episode=ep, streams={})


def test_stub_trainer_satisfies_the_trainer_interface():
    assert isinstance(StubTrainer(), Trainer)


def test_stub_trainer_fits_and_evaluates_per_condition():
    train = [_ep(f"t{i}", ConditionClass.NOMINAL, True) for i in range(4)] + \
            [_ep(f"r{i}", ConditionClass.RUST, False) for i in range(4)]
    eval_set = [_ep("e1", ConditionClass.NOMINAL, True), _ep("e2", ConditionClass.RUST, False)]

    m = StubTrainer().fit(train).evaluate(eval_set)
    assert isinstance(m, TrainMetrics)
    assert m.n_train == 8 and m.n_eval == 2
    assert set(m.per_condition_success) <= {c.value for c in ConditionClass}
    assert all(0.0 <= v <= 1.0 for v in m.per_condition_success.values())
    assert 0.0 <= m.overall_success <= 1.0


def test_stub_trainer_learns_the_condition_majority_baseline():
    # The stub is the condition-majority baseline (the audit's appearance-shortcut reference):
    # nominal always succeeds and rust always fails in train, so predicting each condition's
    # majority scores 1.0 on a matching eval set.
    train = [_ep(f"t{i}", ConditionClass.NOMINAL, True) for i in range(5)] + \
            [_ep(f"r{i}", ConditionClass.RUST, False) for i in range(5)]
    eval_set = [_ep("e1", ConditionClass.NOMINAL, True), _ep("e2", ConditionClass.RUST, False)]

    m = StubTrainer().fit(train).evaluate(eval_set)
    assert m.per_condition_success["nominal"] == 1.0
    assert m.per_condition_success["rust"] == 1.0
    assert m.overall_success == 1.0
