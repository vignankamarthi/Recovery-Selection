"""SIM DRY-RUN of the recovery make-or-break (Part 2, step 2.6) -- NOT a binding gate.

Runs the full recovery loop and reports whether the per-failure oracle beats the best fixed arm by
recovery-regret, and how the calibrated cost-sensitive selector does against both. It then trains and
evaluates the selector under the Lagrangian human budget.

READ THIS BEFORE READING THE NUMBERS. This is MACHINERY validation + a DIRECTIONAL early warning only,
never evidence:
  - the failures are INJECTED (scripted), not organic real-robot failures;
  - the competence tiers are the 1.10 PROXY, not the real ACT latent signals;
  - the outcome model (`recovery.backend.ScriptedRecoveryBackend`) is a stylized, FLAGGED hypothesis
    (mode -> required recovery), so a positive result partly reflects that the machinery computes
    recovery-regret and selects correctly GIVEN the hypothesis, not that recovery works in the world.
The BINDING make-or-break is on hardware (Milestone 3, GATE 4): real ACT signals + real failures.

The number here is computed on the deterministic DATA-LEVEL backend (no sim), so it is reproducible and
fast. The separate sim harness (`tests/recovery/sim_harness.py`) proves the same grid runs on a real
`SimWorld` across the fence; it is not tuned for a favorable number and is not used for this report.

    python scripts/recovery/run_dry_run.py [--n 120] [--seed 0] [--budget 0.30] [--out experiments/...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from recovery.backend import ScriptedRecoveryBackend                    # noqa: E402
from recovery.failures.injection import generate_failures              # noqa: E402
from recovery.grid.counterfactual import CounterfactualGrid            # noqa: E402
from recovery.metric.recovery_regret import DEFAULT_COST_WEIGHTS, RecoveryArm  # noqa: E402
from recovery.selector.selector import CostSensitiveSelector, SelectorFeatures  # noqa: E402
from schema.episode import CompetenceTier                              # noqa: E402

# Which proxy competence tier a failure mode presents as at decision time (the inverse of the tier ->
# mode glue). POSE_PERTURBATION also reads as outside-plannable (its home arm is replan).
from recovery.failures.injection import FailureMode                    # noqa: E402

_MODE_TIER = {
    FailureMode.TRANSIENT_SLIP: CompetenceTier.IN_REGION,
    FailureMode.LABEL_OCCLUSION: CompetenceTier.BOUNDARY,
    FailureMode.POSE_PERTURBATION: CompetenceTier.OUTSIDE_PLANNABLE,
    FailureMode.PLAN_FAILURE: CompetenceTier.OUTSIDE_PLANNABLE,
    FailureMode.UNSAFE_STATE: CompetenceTier.OUTSIDE_RISKY,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--out", type=str, default="experiments/recovery_dry_run.json")
    args = ap.parse_args()

    weights = DEFAULT_COST_WEIGHTS
    failures = generate_failures(n=args.n, seed=args.seed)
    grid = CounterfactualGrid(weights)
    rows = grid.evaluate(lambda f: ScriptedRecoveryBackend(f), failures)

    oracle_mean = grid.mean_oracle_cost(rows)
    best_arm, best_fixed_mean = grid.best_fixed_arm(rows)
    improvement = grid.oracle_improvement(rows)
    fixed = {arm.value: grid.fixed_arm_mean_cost(rows, arm) for arm in RecoveryArm}
    fixed_regret = {arm.value: grid.mean_fixed_arm_regret(rows, arm) for arm in RecoveryArm}

    # Train + calibrate the selector on the (proxy-tier, severity) features.
    samples = [
        (SelectorFeatures(competence_tier=_MODE_TIER[f.mode], severity=f.severity), row)
        for f, row in zip(failures, rows)
    ]
    selector = CostSensitiveSelector(weights, human_budget=args.budget).fit(samples)
    lam = selector.calibrate(samples)
    report = selector.evaluate(samples)

    result = {
        "caveat": "DIRECTIONAL / MACHINERY ONLY -- not evidence. Injected failures + proxy tiers + a "
                  "stylized outcome model. The binding make-or-break is hardware (GATE 4).",
        "n_failures": len(failures),
        "cost_weights": weights.__dict__,
        "human_budget": args.budget,
        "oracle_mean_cost": round(oracle_mean, 3),
        "best_fixed_arm": best_arm.value,
        "best_fixed_mean_cost": round(best_fixed_mean, 3),
        "oracle_improvement_over_best_fixed": round(improvement, 4),
        "oracle_beats_best_fixed": bool(oracle_mean < best_fixed_mean),
        "fixed_arm_mean_cost": {k: round(v, 3) for k, v in fixed.items()},
        "fixed_arm_mean_regret": {k: round(v, 3) for k, v in fixed_regret.items()},
        "selector": {
            "lambda": round(lam, 3),
            "mean_cost": round(report.mean_cost, 3),
            "mean_regret": round(report.mean_regret, 3),
            "ask_human_fraction": round(report.ask_human_fraction, 4),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
