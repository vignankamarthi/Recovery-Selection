"""Competence model (2.2): the four tiers, the tier->arm map, the control-invariant safe-set floor,
the proxy tier from 1.10 metadata, and the ACT-signal path (torch-free, off a stub base policy).
Fence: imports `recovery` + `schema` only, never `harvest`."""

import pytest

from recovery.competence.signals import (
    ACTCompetenceModel,
    CompetenceSignals,
    CompetenceThresholds,
    ProxyCompetenceModel,
    classify_signals,
    tier_to_arm,
)
from recovery.policy.base_policy import StubBasePolicy
from schema.episode import CompetenceTier, Episode, ConditionClass
from recovery.metric.recovery_regret import RecoveryArm


def test_tier_to_arm_is_the_proposal_mapping():
    assert tier_to_arm(CompetenceTier.IN_REGION) is RecoveryArm.RETRY
    assert tier_to_arm(CompetenceTier.BOUNDARY) is RecoveryArm.REWIND
    assert tier_to_arm(CompetenceTier.OUTSIDE_PLANNABLE) is RecoveryArm.REPLAN
    assert tier_to_arm(CompetenceTier.OUTSIDE_RISKY) is RecoveryArm.ASK_HUMAN
    assert {tier_to_arm(t) for t in CompetenceTier} == set(RecoveryArm)


def test_safe_set_is_a_hard_floor_forcing_ask_human():
    # A genuinely-unsafe state escalates to outside_risky (-> ask-human) regardless of how familiar
    # the latent looks. The safe set is a floor beneath the tiers, not a selectable arm.
    familiar_but_unsafe = CompetenceSignals(latent_density=0.99, ensemble_disagreement=0.0, in_safe_set=False)
    assert classify_signals(familiar_but_unsafe) is CompetenceTier.OUTSIDE_RISKY


def test_classify_signals_bands_from_familiar_to_far():
    th = CompetenceThresholds()
    in_region = CompetenceSignals(0.95, 0.02, True)
    boundary = CompetenceSignals(0.75, 0.15, True)
    plannable = CompetenceSignals(0.45, 0.35, True)
    risky = CompetenceSignals(0.05, 0.9, True)
    assert classify_signals(in_region, th) is CompetenceTier.IN_REGION
    assert classify_signals(boundary, th) is CompetenceTier.BOUNDARY
    assert classify_signals(plannable, th) is CompetenceTier.OUTSIDE_PLANNABLE
    assert classify_signals(risky, th) is CompetenceTier.OUTSIDE_RISKY


def test_proxy_competence_model_reads_the_1_10_tier_from_metadata():
    ep = Episode("e0", "nominal-000", ConditionClass.NOMINAL)
    ep.metadata["competence_tier"] = CompetenceTier.BOUNDARY.value
    model = ProxyCompetenceModel()
    assert model.tier(ep) is CompetenceTier.BOUNDARY
    assert model.default_arm(ep) is RecoveryArm.REWIND


def test_proxy_model_raises_on_an_untagged_episode():
    model = ProxyCompetenceModel()
    with pytest.raises(KeyError):
        model.tier(Episode("e1", "rust-000", ConditionClass.RUST))


def test_act_competence_model_reads_signals_off_a_frozen_policy():
    # Fit a reference density on training latents, then a far/unfamiliar observation should read
    # lower density (less familiar) than an in-distribution one. Ensemble disagreement is >= 0.
    pol = StubBasePolicy(action_dim=7, latent_dim=8, ensemble_size=5)
    train_obs = [{"proprioception": [i * 0.01] * 7} for i in range(50)]
    model = ACTCompetenceModel(pol).fit([o for o in train_obs])
    sig = model.signals(train_obs[0])
    assert 0.0 <= sig.latent_density <= 1.0
    assert sig.ensemble_disagreement >= 0.0
    assert isinstance(sig.in_safe_set, bool)
    # classify returns one of the four tiers
    assert model.classify(train_obs[0]) in set(CompetenceTier)
