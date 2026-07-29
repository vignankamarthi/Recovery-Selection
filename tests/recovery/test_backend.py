"""The abstract recovery backend (the seam the arms drive) + the schema-only reference outcome model
`ScriptedRecoveryBackend`. The reference backend encodes the FLAGGED mode->required-recovery
hypothesis so the grid/arms/selector are testable with no sim. Fence: imports `recovery` + `schema`."""

from recovery.backend import RecoveryBackend, ScriptedRecoveryBackend
from recovery.failures.injection import FailureMode, InjectedFailure
from schema.episode import ConditionClass


def _failure(mode, severity=0.5, seed=0):
    return InjectedFailure("f", mode, ConditionClass.RUST, severity, seed)


def test_scripted_backend_is_a_recovery_backend_structurally():
    assert isinstance(ScriptedRecoveryBackend(_failure(FailureMode.TRANSIENT_SLIP)), RecoveryBackend)


def test_snapshot_restore_resets_the_attempt_state():
    b = ScriptedRecoveryBackend(_failure(FailureMode.TRANSIENT_SLIP))
    snap = b.snapshot()
    b.perturb(0.1)
    b.present()
    assert b.elapsed_s() > 0
    b.restore(snap)
    assert b.elapsed_s() == 0.0            # a fresh attempt from the same failure state


def test_transient_slip_is_recovered_by_a_retry_level_action():
    # perturb + present (the retry vocabulary) recovers a transient slip.
    b = ScriptedRecoveryBackend(_failure(FailureMode.TRANSIENT_SLIP))
    b.perturb(0.1)
    b.present()
    assert b.task_success() is True


def test_occlusion_needs_a_rewind_not_a_bare_retry():
    # A retry-level action does NOT clear an occlusion; retreat + re-approach + present does.
    retry = ScriptedRecoveryBackend(_failure(FailureMode.LABEL_OCCLUSION))
    retry.perturb(0.1)
    retry.present()
    assert retry.task_success() is False

    rewind = ScriptedRecoveryBackend(_failure(FailureMode.LABEL_OCCLUSION))
    rewind.retreat()
    rewind.reapproach()
    rewind.present()
    assert rewind.task_success() is True


def test_plan_failure_needs_a_replan():
    b = ScriptedRecoveryBackend(_failure(FailureMode.PLAN_FAILURE))
    b.retreat()
    b.reapproach()
    b.present()
    assert b.task_success() is False        # rewind is not enough
    b2 = ScriptedRecoveryBackend(_failure(FailureMode.PLAN_FAILURE))
    b2.retreat()
    assert b2.replan() is True
    b2.reapproach()
    b2.present()
    assert b2.task_success() is True


def test_unsafe_state_only_a_human_recovers_and_autonomous_action_trips_the_safe_set():
    autonomous = ScriptedRecoveryBackend(_failure(FailureMode.UNSAFE_STATE))
    assert autonomous.is_safe() is True     # safe until something acts
    autonomous.perturb(0.1)
    assert autonomous.is_safe() is False    # autonomous action in an unsafe failure trips the floor
    autonomous.present()
    assert autonomous.task_success() is False

    human = ScriptedRecoveryBackend(_failure(FailureMode.UNSAFE_STATE))
    human.request_human()
    assert human.task_success() is True
    assert human.is_safe() is True          # deferring to a human never trips the safe set
