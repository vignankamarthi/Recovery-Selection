"""Failure-injection API (2.3): a deterministic scripted catalog of 100-150 failures across the can
tasks, on top of the organic condition/orientation failures. Backend-agnostic (schema-only): each
`InjectedFailure` DESCRIBES a failure; a backend (data-level stub or the sim harness) applies it.
Fence: imports `recovery` + `schema` only."""

from recovery.failures.injection import (
    FailureMode,
    InjectedFailure,
    generate_failures,
)
from schema.episode import ConditionClass


def test_catalog_size_is_in_the_proposal_range():
    failures = generate_failures()
    assert 100 <= len(failures) <= 150


def test_catalog_is_deterministic_under_seed():
    a = generate_failures(n=120, seed=7)
    b = generate_failures(n=120, seed=7)
    c = generate_failures(n=120, seed=8)
    assert a == b
    assert a != c


def test_every_mode_and_every_condition_appears():
    failures = generate_failures(n=140, seed=0)
    assert {f.mode for f in failures} == set(FailureMode)
    assert {f.condition for f in failures} == set(ConditionClass)


def test_severity_is_a_unit_interval_and_ids_are_unique():
    failures = generate_failures(n=130, seed=1)
    assert all(0.0 <= f.severity <= 1.0 for f in failures)
    assert len({f.failure_id for f in failures}) == len(failures)


def test_injected_failure_is_frozen_and_carries_a_seed():
    f = InjectedFailure(
        failure_id="f0", mode=FailureMode.TRANSIENT_SLIP,
        condition=ConditionClass.RUST, severity=0.5, seed=3,
    )
    assert f.mode is FailureMode.TRANSIENT_SLIP
    assert f.seed == 3
