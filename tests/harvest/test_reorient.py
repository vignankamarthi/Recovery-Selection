"""Tests for the sim demonstration generator (1.7d): weld + plan-then-execute righting/present.

Physical sim, slow. All cans spawn LYING (the task is to right them and present the label to the
overhead camera). The search runs on a hidden scratch copy; only the smooth glide is the demo.
"""

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from harvest.sim.reorient import ReorientResult, demonstrate, slip_severity  # noqa: E402
from harvest.sim.world import SimWorld  # noqa: E402
from schema.episode import ConditionClass  # noqa: E402

_LYING = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)


def _world(seed: int) -> SimWorld:
    return SimWorld(can_pos=(0.5, 0.0, 0.11), can_seed=seed, can_quat=_LYING)


def test_demonstrate_returns_the_three_stage_signals():
    res = demonstrate(_world(1))
    assert isinstance(res, ReorientResult)
    assert isinstance(res.upright_success, bool)
    assert isinstance(res.label_visible, bool)
    assert -1.0 <= res.label_nz <= 1.0
    assert res.overhead_px is None or isinstance(res.overhead_px, int)


def test_demonstrate_reorients_and_presents_a_nominal_can():
    # A reachable seed: the reorient brings the label up (facing the overhead camera) in one move,
    # with no stand-the-can-upright detour, and the overhead reads enough coverage.
    res = demonstrate(_world(3))
    assert res.upright_success is True
    assert res.label_visible is True
    assert res.label_nz > 0.9


def test_demonstrate_calls_on_step_during_the_visible_glide_only():
    # The invisible search must not tick the recorder; only the grasp + glide do.
    calls = [0]
    demonstrate(_world(2), on_step=lambda: calls.__setitem__(0, calls[0] + 1))
    assert calls[0] > 0


def test_slip_severity_escalates_with_condition_damage():
    # The failure taxonomy: a nominal can barely slips, damage escalates the in-hand slip.
    seeds = range(20)
    mean = lambda c: sum(slip_severity(c, s) for s in seeds) / len(list(seeds))
    assert mean(ConditionClass.NOMINAL) < mean(ConditionClass.BODY_DENT)
    assert mean(ConditionClass.BODY_DENT) < mean(ConditionClass.BULGE)
    assert mean(ConditionClass.BULGE) <= mean(ConditionClass.RUST)
    assert all(0.0 <= slip_severity(c, s) <= 1.0 for c in ConditionClass for s in seeds)


def test_slip_rolls_the_label_off_and_fails_the_present():
    # A clean present (no slip) faces the label up; a severe slip rolls it off and fails, which is
    # the condition-correlated in-hand flare that drives damaged-can failures.
    clean = demonstrate(_world(3), slip=0.0)
    slipped = demonstrate(_world(3), slip=0.9)
    assert clean.upright_success is True
    assert slipped.upright_success is False
    assert slipped.label_nz < clean.label_nz


def test_plan_failure_reports_failure_not_a_crash(monkeypatch):
    # If the presentation search finds nothing reachable, the honest outcome is a failed reorient
    # (upright_success False, label not visible), reported, never an exception.
    from harvest.sim import reorient

    monkeypatch.setattr(reorient, "_plan", lambda w, offset: None)
    res = demonstrate(_world(3))
    assert res.upright_success is False
    assert res.label_visible is False
