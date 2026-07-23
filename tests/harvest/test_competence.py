"""Tests for proxy competence-tier tagging (1.10). The tier comes from the HELD-OUT realized margin
(`label_up_cos` + `label_visible`), never the generative condition, so the recovery smoke test and
the GATE-4 probe are not circular. Pure metadata, no sim, no streams."""

from harvest.dataset.competence import competence_tier, tag_competence
from schema.episode import (
    CompetenceTier,
    ConditionClass,
    Episode,
    Label,
    LabelProvenance,
    Outcome,
)


def _ep(cond: ConditionClass, cos: float, visible: bool, eid: str = "e") -> Episode:
    outcome = Outcome.SUCCESS if (cos >= 0.92 and visible) else Outcome.FAILURE
    return Episode(eid, f"{cond.value}-{eid}", cond, outcome=outcome, labels=[
        Label("label_up_cos", cos, LabelProvenance.AUTO_VISION),
        Label("label_visible", visible, LabelProvenance.AUTO_VISION),
    ])


def test_tier_bins_by_realized_margin():
    assert competence_tier(_ep(ConditionClass.NOMINAL, 0.99, True)) is CompetenceTier.IN_REGION
    assert competence_tier(_ep(ConditionClass.BODY_DENT, 0.90, False)) is CompetenceTier.BOUNDARY
    assert competence_tier(_ep(ConditionClass.SEAM_DENT, 0.78, False)) is CompetenceTier.OUTSIDE_PLANNABLE
    assert competence_tier(_ep(ConditionClass.RUST, 0.45, False)) is CompetenceTier.OUTSIDE_RISKY


def test_coverage_failure_is_not_in_region_despite_high_orientation():
    # A can that faces up (high cos) but is occluded from the overhead camera (not visible) is a
    # near-miss coverage failure, boundary, not in-region (audit finding R5).
    assert competence_tier(_ep(ConditionClass.NOMINAL, 0.97, False)) is CompetenceTier.BOUNDARY


def test_tier_is_held_out_from_condition():
    # Same margin, different condition -> same tier. The tier must not read the generative variable.
    a = competence_tier(_ep(ConditionClass.NOMINAL, 0.80, False))
    b = competence_tier(_ep(ConditionClass.RUST, 0.80, False))
    assert a is b


def test_tag_competence_writes_the_tier_into_episode_metadata():
    eps = [_ep(ConditionClass.NOMINAL, 0.99, True, "a"), _ep(ConditionClass.RUST, 0.40, False, "b")]
    tag_competence(eps)
    assert eps[0].metadata["competence_tier"] == CompetenceTier.IN_REGION.value
    assert eps[1].metadata["competence_tier"] == CompetenceTier.OUTSIDE_RISKY.value
    # all four tiers are valid schema values (the recovery layer maps them to the four arms)
    assert {t.value for t in CompetenceTier} == {
        "in_region", "boundary", "outside_plannable", "outside_risky"
    }
