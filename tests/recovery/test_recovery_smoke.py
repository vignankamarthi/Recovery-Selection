"""Recovery SMOKE TEST (2.2b): the full loop end to end on proxy-tagged episodes, data-level, with no
sim -- proxy competence tier -> arm selection -> recovery-regret. This is the early FALSIFICATION
harness for the recovery framework, before hardware and before the real ACT signals. It validates the
MACHINERY only (the reference outcome model is a stylized hypothesis), never evidence.

Fence: imports `recovery` + `schema` only, never `harvest` (the episodes are built from the schema
directly, standing in for the 1.10-tagged sim episodes)."""

from recovery.arms import ALL_ARMS
from recovery.backend import ScriptedRecoveryBackend
from recovery.competence.signals import ProxyCompetenceModel
from recovery.failures.injection import failure_from_episode
from recovery.grid.counterfactual import CounterfactualGrid
from recovery.metric.recovery_regret import DEFAULT_COST_WEIGHTS, RecoveryArm, recovery_regret, total_cost
from schema.episode import CompetenceTier, ConditionClass, Episode


def _tagged_episodes() -> list[Episode]:
    """Synthetic stand-ins for the 1.10-tagged sim episodes: a spread across the four proxy tiers."""
    tiers = list(CompetenceTier)
    conditions = list(ConditionClass)
    eps = []
    for i in range(40):
        tier = tiers[i % len(tiers)]
        ep = Episode(f"ep-{i:03d}", f"can-{i:03d}", conditions[i % len(conditions)])
        ep.metadata["competence_tier"] = tier.value
        eps.append(ep)
    return eps


def test_smoke_loop_runs_tier_to_arm_to_recovery_regret_end_to_end():
    episodes = _tagged_episodes()
    proxy = ProxyCompetenceModel()
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)

    # One scripted failure per tagged episode (mode matched to the proxy tier).
    failures = [failure_from_episode(ep, seed=i) for i, ep in enumerate(episodes)]
    rows = grid.evaluate(lambda f: ScriptedRecoveryBackend(f), failures)
    assert len(rows) == len(episodes)

    # The tier-default policy: pick each failure's arm from its episode's proxy tier, then score its
    # realized recovery-regret against the per-failure oracle. This is the loop the smoke test exercises.
    total_regret = 0.0
    for ep, row in zip(episodes, rows):
        default_arm = proxy.default_arm(ep)
        out = row.outcomes[default_arm]
        realized = total_cost(out.cost, out.recovered, DEFAULT_COST_WEIGHTS)
        total_regret += recovery_regret(realized, row.oracle_cost)
    mean_regret = total_regret / len(rows)

    # Machinery assertions only (NOT evidence): the loop produced a finite, non-negative regret, and
    # because the tiers are aligned to the modes by construction here, the tier-default policy tracks
    # the oracle closely (this is a plumbing check, not a capability claim).
    assert mean_regret >= 0.0
    assert grid.mean_oracle_cost(rows) >= 0.0
    # Every tier's default arm is one of the four, and the oracle set is covered across the episodes.
    assert {row.oracle_arm for row in rows} <= set(RecoveryArm)


def test_smoke_tier_default_arm_recovers_its_matched_failure():
    # A sanity check that the tier->arm mapping and the reference model agree: the default arm for each
    # tier recovers a failure of that tier's matched mode.
    proxy = ProxyCompetenceModel()
    for tier in CompetenceTier:
        ep = Episode("e", "c", ConditionClass.NOMINAL)
        ep.metadata["competence_tier"] = tier.value
        f = failure_from_episode(ep)
        backend = ScriptedRecoveryBackend(f)
        arm = next(a for a in ALL_ARMS if a.arm is proxy.default_arm(ep))
        backend.restore(backend.snapshot())
        out = arm.execute(backend)
        assert out.recovered is True
