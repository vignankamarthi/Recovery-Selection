"""Proxy competence-tier tagging (Part 1, step 1.10) -- the sim's double duty.

Tags each episode with one of the four `CompetenceTier`s so the recovery smoke test (2.2b) can run
the tier -> arm -> recovery-regret loop on the same episodes that train ACT, before the real ACT
latent signals exist. The tier is a PROXY read from the HELD-OUT realized margin (the presented
label's orientation cosine `label_up_cos` and whether the overhead camera saw it `label_visible`),
NEVER from the generative variables (condition / orientation / deformation) that created the
failure. Reading the generative variable would make the recovery smoke test and the GATE-4
separability probe circular. When the real ACT trains (1.11), its latent-state density + ensemble
disagreement replace this proxy.

Fence-safe: this writes a `schema` enum value into `Episode.metadata`. `recovery` reads that
metadata, it never imports `harvest`, and `harvest` never imports `recovery`.
"""

from __future__ import annotations

from typing import Iterable, Optional

from schema.episode import CompetenceTier, Episode

# Margin bands on `label_up_cos` (the realized presentation margin; the success threshold is 0.92).
# Comfortable success is in-region, straddling the threshold is boundary, a clear miss is
# replannable, a large miss is risky. Proxy bands, superseded by the real ACT latent signal.
_IN_REGION_MIN = 0.95
_BOUNDARY_MIN = 0.88
_PLANNABLE_MIN = 0.70


def _label(episode: Episode, name: str, default: Optional[object] = None) -> object:
    for lab in episode.labels:
        if lab.name == name:
            return lab.value
    return default


def competence_tier(episode: Episode) -> CompetenceTier:
    """The proxy tier for one episode, from its held-out realized margin (never its condition)."""
    cos = float(_label(episode, "label_up_cos", 0.0))
    visible = bool(_label(episode, "label_visible", False))
    if cos >= _IN_REGION_MIN and visible:
        return CompetenceTier.IN_REGION          # faced up with room to spare and the camera saw it
    if cos >= _BOUNDARY_MIN:
        return CompetenceTier.BOUNDARY           # near the edge either way (marginal, or occluded)
    if cos >= _PLANNABLE_MIN:
        return CompetenceTier.OUTSIDE_PLANNABLE  # a clear miss, still recoverable by a replan
    return CompetenceTier.OUTSIDE_RISKY          # rolled far off, ask a human


def tag_competence(episodes: Iterable[Episode]) -> None:
    """Write each episode's proxy tier into `episode.metadata['competence_tier']` (in place)."""
    for episode in episodes:
        episode.metadata["competence_tier"] = competence_tier(episode).value
