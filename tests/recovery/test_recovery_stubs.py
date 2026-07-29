"""Fence + enum contract for Part 2 (recovery). Importing `recovery` alone exercises the fence
(recovery imports schema only, never harvest). The per-module behavior is covered by the dedicated
test files (test_metric, test_base_policy, test_competence_model, test_injection, test_backend,
test_arms, test_grid, test_recovery_smoke, test_selector)."""

from recovery.metric.recovery_regret import RecoveryArm


def test_the_four_recovery_arms_exist():
    # The escalation ladder Part 2 selects over (ask-human is the terminal safety fallback).
    assert {a.value for a in RecoveryArm} == {"retry", "rewind", "replan", "ask_human"}
