"""The four recovery arms (2.4) as backend-agnostic behaviors: each drives the `RecoveryBackend`
vocabulary and returns an `ArmOutcome` (arm, cost, recovered). One arm per competence tier. Fence:
imports `recovery` + `schema` only."""

from recovery.arms import ALL_ARMS, Arm, ArmOutcome, AskHumanArm, ReplanArm, RetryArm, RewindArm
from recovery.backend import ScriptedRecoveryBackend
from recovery.failures.injection import FailureMode, InjectedFailure
from recovery.metric.recovery_regret import RecoveryArm
from schema.episode import ConditionClass


def _f(mode, severity=0.5):
    return InjectedFailure("f", mode, ConditionClass.RUST, severity, 0)


def test_all_four_arms_are_present_in_ladder_order():
    assert [a.arm for a in ALL_ARMS] == [
        RecoveryArm.RETRY, RecoveryArm.REWIND, RecoveryArm.REPLAN, RecoveryArm.ASK_HUMAN,
    ]
    for a in ALL_ARMS:
        assert isinstance(a, Arm)


def test_retry_recovers_a_transient_slip_cheaply():
    out = RetryArm().execute(ScriptedRecoveryBackend(_f(FailureMode.TRANSIENT_SLIP)))
    assert isinstance(out, ArmOutcome)
    assert out.arm is RecoveryArm.RETRY
    assert out.recovered is True
    assert out.cost.human_effort == 0.0
    assert out.cost.time_s > 0


def test_retry_fails_on_an_occlusion_but_rewind_recovers_it():
    assert RetryArm().execute(ScriptedRecoveryBackend(_f(FailureMode.LABEL_OCCLUSION))).recovered is False
    assert RewindArm().execute(ScriptedRecoveryBackend(_f(FailureMode.LABEL_OCCLUSION))).recovered is True


def test_replan_recovers_a_plan_failure():
    assert ReplanArm().execute(ScriptedRecoveryBackend(_f(FailureMode.PLAN_FAILURE))).recovered is True


def test_ask_human_always_recovers_and_spends_human_effort_without_safety_violations():
    out = AskHumanArm().execute(ScriptedRecoveryBackend(_f(FailureMode.UNSAFE_STATE)))
    assert out.arm is RecoveryArm.ASK_HUMAN
    assert out.recovered is True
    assert out.cost.human_effort > 0
    assert out.cost.safety_violations == 0


def test_autonomous_arm_on_an_unsafe_state_accrues_safety_violations():
    # The retry arm acting on a genuinely-unsafe failure trips the safe-set floor (and fails to recover).
    out = RetryArm().execute(ScriptedRecoveryBackend(_f(FailureMode.UNSAFE_STATE)))
    assert out.recovered is False
    assert out.cost.safety_violations >= 1


def test_time_cost_escalates_along_the_ladder():
    # Same failure, the more elaborate arms spend more time.
    f = FailureMode.TRANSIENT_SLIP
    t_retry = RetryArm().execute(ScriptedRecoveryBackend(_f(f))).cost.time_s
    t_rewind = RewindArm().execute(ScriptedRecoveryBackend(_f(f))).cost.time_s
    t_replan = ReplanArm().execute(ScriptedRecoveryBackend(_f(f))).cost.time_s
    assert t_retry < t_rewind < t_replan
