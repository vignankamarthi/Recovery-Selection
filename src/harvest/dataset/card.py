"""Dataset card + split viz (Phase 1.8).

A human-facing summary of the dataset with the honest sim caveats, plus a text table of
the condition-by-split can counts. Dependency-light (stdlib only) so it runs anywhere.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from harvest.dataset.splits import Split
from schema.episode import ConditionClass, Episode, Outcome

_SPLIT_ORDER = ("train", "val", "test")


def _split_of_can(split: Split) -> dict[str, str]:
    out: dict[str, str] = {}
    for cid in split.train_can_ids:
        out[cid] = "train"
    for cid in split.val_can_ids:
        out[cid] = "val"
    for cid in split.test_can_ids:
        out[cid] = "test"
    return out


def dataset_stats(episodes: Iterable[Episode], split: Split) -> dict:
    """Summary counts over the dataset, keyed for the card and the viz table."""
    episodes = list(episodes)
    cond_by_can = {e.can_id: e.condition for e in episodes}
    split_of_can = _split_of_can(split)

    per_cond_total = Counter(e.condition.value for e in episodes)
    per_cond_success = Counter(
        e.condition.value for e in episodes if e.outcome is Outcome.SUCCESS
    )
    success_rate_per_condition = {
        c: (per_cond_success[c] / per_cond_total[c] if per_cond_total[c] else 0.0)
        for c in per_cond_total
    }

    matrix: dict[str, Counter] = defaultdict(Counter)
    for cid, cond in cond_by_can.items():
        matrix[cond.value][split_of_can.get(cid, "?")] += 1

    return {
        "total_episodes": len(episodes),
        "total_cans": len(cond_by_can),
        "cans_per_condition": dict(Counter(c.value for c in cond_by_can.values())),
        "cans_per_split": dict(Counter(split_of_can.get(cid, "?") for cid in cond_by_can)),
        "outcomes": dict(Counter(e.outcome.value for e in episodes if e.outcome is not None)),
        "success_rate_per_condition": success_rate_per_condition,
        "condition_split_matrix": {k: dict(v) for k, v in matrix.items()},
        "leak_free": split.is_leak_free(),
    }


def render_split_table(episodes: Iterable[Episode], split: Split) -> str:
    """A text table of can counts, condition rows by train/val/test columns."""
    stats = dataset_stats(episodes, split)
    matrix = stats["condition_split_matrix"]
    header = f"{'condition':<12}" + "".join(f"{c:>7}" for c in _SPLIT_ORDER) + f"{'total':>8}"
    lines = [header, "-" * len(header)]
    col_totals = [0, 0, 0]
    for cond in (c.value for c in ConditionClass):
        row = matrix.get(cond, {})
        counts = [row.get(c, 0) for c in _SPLIT_ORDER]
        for i, n in enumerate(counts):
            col_totals[i] += n
        lines.append(
            f"{cond:<12}" + "".join(f"{n:>7}" for n in counts) + f"{sum(counts):>8}"
        )
    lines.append(
        f"{'ALL':<12}" + "".join(f"{n:>7}" for n in col_totals) + f"{sum(col_totals):>8}"
    )
    return "\n".join(lines)


def dataset_card(episodes: Iterable[Episode], split: Split) -> str:
    """A markdown dataset card, including the by-can leak-free statement and sim caveats."""
    episodes = list(episodes)
    s = dataset_stats(episodes, split)
    conds = ", ".join(c.value for c in ConditionClass)

    lines = [
        "# HARVEST canned-goods manipulation dataset (synthetic, MuJoCo sim)",
        "",
        f"- Episodes: {s['total_episodes']}",
        f"- Distinct physical cans: {s['total_cans']}",
        f"- Condition classes: {conds}",
        f"- Splits are BY CAN and leak-free: {s['leak_free']} "
        "(no can_id appears in two splits, HARVEST flag F6)",
        "",
        "## Cans per condition",
    ]
    lines += [f"- {c.value}: {s['cans_per_condition'].get(c.value, 0)}" for c in ConditionClass]
    lines += ["", "## Cans per split"]
    lines += [f"- {k}: {s['cans_per_split'].get(k, 0)}" for k in _SPLIT_ORDER]
    lines += ["", "## Condition by split (can counts)", "```", render_split_table(episodes, split), "```"]
    lines += ["", "## Episode success rate per condition"]
    for c in ConditionClass:
        r = s["success_rate_per_condition"].get(c.value)
        lines.append(f"- {c.value}: {r:.2f}" if r is not None else f"- {c.value}: n/a")
    lines += [
        "",
        "## Sim caveats (read before using)",
        "- This is a MuJoCo synthetic dataset, not real-robot data. A sim-to-real gap applies. It exists to validate the recording and training pipeline, NOT as evidence about the real sensor or the real task. The reported dataset, ACT baseline, and tactile ablation all come from the PHYSICAL dataset.",
        "- The in-hand reorient is KINEMATICALLY WELDED. The sim's rigid finger-pad friction cannot hold a can through a reorient, so the can is force-attached to the hand for that phase. Contact and grasp physics during the reorient are therefore NOT real.",
        "- `grasp_stable` is a CONSTANT SIMULATOR DEFAULT (always true), not a measured signal. Because the weld carries the can, there is no honest hold signal to read in sim. It carries ZERO information in this dataset. Do not train on it or report it. It becomes a real tactile/physics signal only on hardware.",
        "- FAILURES ARE INJECTED, not emergent. Each can gets a condition-scaled in-hand SLIP (a roll about the can's long axis) that rolls the nutrition label off the top so the overhead read fails. Severity is a condition base plus a per-can seeded jitter, so the failure gradient is correlated with damage BY CONSTRUCTION, not discovered by physics. Treat the per-condition success rates as a designed distribution, not a measurement.",
        "- Tactile is a spatial-binning proxy, not the real 28-taxel TSF-85 taxel layout. Do not read physical taxel geometry from it. It is spatially structured only during the brief real grasp; through the welded reorient the kinematically-placed can yields a near-uniform (roughly 2-value) map, so the sim tactile channel cannot support a meaningful tactile-vs-vision comparison. The sim ablation is a pipeline smoke test only.",
        "- Condition deformations are rigid geometric variants. Convex geoms cannot form a true concave dent.",
        "- No label is derived from the tactile stream (HARVEST flag F7), so the tactile ablation is not self-confounded.",
    ]
    return "\n".join(lines)
