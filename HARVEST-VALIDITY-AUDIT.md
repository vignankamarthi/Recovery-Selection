# HARVEST-Touch sim dataset -- validity audit (step 1.9b)

Red-team / blue-team audit of the condition-aware pick-and-reorient task and the generated sim
dataset against the proposal. Dated 2026-07-22. No hard stop at this step. The finding is a
recommendation to proceed with the documented default by-can split, flagged for ratification at
GATE 3 (the ACT baseline review), per PLAN.md.

## Dataset under audit

`data/harvest_sim_v1/` (regenerable via `scripts/cluster/gen_dataset_parallel.py`). 600 episodes over 200 distinct
cans (5 conditions x 40 cans x 3 poses). The Mac holds metadata only, the heavy per-episode streams
are materialized on the cluster at ACT time. Three graded labels plus the held-out `label_up_cos`
margin. All numbers below come from `metadata.jsonl`.

The task realized in the dataset is this. A can spawns lying, the robot grasps it by the head, reorients
it in hand directly to horizontal with the nutrition label facing the fixed overhead camera, and
the overhead read plus the reorient geometry decide the outcome. Failures are a condition-scaled
in-hand slip. This matches the proposal's grasp -> in-hand reorient -> expose-label -> verify
episode, with the deliberate, documented deviations (all-lying spawn, head grasp, direct reorient
with no stand-upright step).

## Blue team -- what holds

1. **By-can split is leak-free.** 0 cans appear in more than one split. The independence unit is
   the can (F5/F6), and no can's geometry is seen in both train and evaluation.
2. **Splits are condition-stratified and balanced.** Every split carries all five conditions
   (train 28 cans each, val and test 6 cans each). Failure rate is even across splits (train
   0.56, val 0.61, test 0.61), so no split is accidentally all-easy or all-hard.
3. **Outcome is not degenerate.** Every condition contains both outcomes, and 73 of 200 cans flip
   outcome across their three poses, so the label is not a pure per-can constant.
4. **The failure gradient is the intended one and monotonic.** Success rate falls nominal 0.88,
   body_dent 0.59, seam_dent 0.45, bulge 0.10, rust 0.08, so the dataset encodes the intended
   damage-correlated failure the recovery layer consumes.
5. **The pipeline round-trips over the full run.** Generate -> per-episode persist -> by-can split
   -> HF metadata + card ran over all 600 episodes in parallel shards, which is the
   recording-pipeline validation the sim exists to provide.

## Red team -- threats

Ordered by severity. Each notes whether it is sim-specific (does not touch the reported hardware
result) or general.

### R1. Static-appearance shortcut (sim-specific, the main finding)

Outcome is highly predictable from condition alone. A predictor that sees only the condition and
guesses that condition's majority outcome scores 0.768, against a 0.578 base rate. Condition is
visible in RGB (dent shape, bulge cap, rust color), so a vision model can approach that 0.77
bound from the can's static appearance, without observing the slip and without tactile. This
inflates vision-only performance and shrinks the apparent tactile-vs-vision gap in sim.

This is expected and acceptable, because the sim tactile ablation is a smoke test only and is
never reported. On hardware the slip is stochastic and not fixed by appearance, so tactile
carries information appearance cannot. The threat is a reason the reported ablation must be the
physical one, which the proposal already requires. Mitigation. keep the sim ablation labeled a
smoke test, report the appearance-shortcut bound (0.81) alongside any sim vision-only number so it
is never read as evidence.

### R2. Outcome is largely a per-can property (general, mild)

The slip severity is deterministic per can, so 127 of 200 cans share one outcome across all three
poses. Effective independent outcome units are near 200 cans, not 600 episodes, and the three
poses of a can are correlated. The by-can split already prevents this from leaking, and the 73
mixed cans show pose adds real variation. Mitigation. report metrics per-can where it matters, and
state the effective sample size (about 200 cans) rather than 600 episodes.

### R3. Failure is a late-trajectory event (general, mild)

The slip is applied after the reorient glide, so a success and a failure episode are identical
until the final slip. Early-trajectory streams carry no failure signal. For ACT action imitation
this is fine. For a failure detector or the competence signal it means the cue is late. On
hardware slip onset is gradual and tactile-detectable earlier, which is again where tactile helps.
Mitigation. the 1.10 competence signal should use the realized margin, not early frames.

### R4. `grasp_stable` carries zero information (sim-specific, disclosed)

`grasp_stable` is a constant sim default (always true), because the weld carries the can and there
is no honest hold signal to read in sim. Already disclosed in the dataset card. Mitigation. do not
train on it or report it, treat it as a hardware-only signal.

### R5. `label_up_cos` margin overlaps the two failure modes (general, mild)

The outcome has two gates, orientation (`upright_success`, nz > 0.92) and coverage
(`label_visible`, overhead px). Some failures keep a high `label_up_cos` because they failed on
coverage, not orientation, so the margin alone does not cleanly separate success from failure.
Mitigation. the 1.10 tier should read both the orientation margin and the coverage margin, not
`label_up_cos` alone.

### R6. Small evaluation subsets (sim-specific, scale)

Val and test carry 6 cans per condition, so a per-condition damaged-subset metric on a split rests
on 18 episodes. This is the smoke-test scale. The proposal's acceptance criteria (about 50
cans/class, about 10 episodes/can) are targets for the physical dataset, not this sim set.
Mitigation. report sim per-condition numbers with the small-N caveat and do not over-read them.

## Threat summary

| Id | Threat | Severity | Sim-specific | Mitigation in place |
|----|--------|----------|--------------|---------------------|
| R1 | Static-appearance shortcut (0.77 vs 0.58 base) | Medium | Yes | Sim ablation is a smoke test, report the bound |
| R2 | Outcome largely per-can (127/200 homogeneous) | Low | No | By-can split, report ESS ~200 cans |
| R3 | Failure is a late event | Low | No | 1.10 uses the realized margin |
| R4 | `grasp_stable` is constant | Low | Yes | Disclosed, do not train or report it |
| R5 | `label_up_cos` overlaps failure modes | Low | No | 1.10 reads orientation and coverage |
| R6 | Small val/test subsets | Low | Yes | Physical dataset meets the scale target |

## Verdict

Proceed with the documented default by-can split. The split is leak-free, balanced, and
representative, and the task realized in the dataset is the proposal's task. The one medium threat
(R1) is sim-specific and consistent with the sim being a smoke test, so it does not touch the
reported hardware result. The dataset is valid for its purpose, validating the recording, ACT, and
recovery pipelines before hardware.

## For GATE 3 to ratify

- The by-can split policy (default, failure-heavy balancing left off).
- Reporting any sim vision-only number next to the 0.77 appearance-shortcut bound, never as
  standalone evidence.
- The effective-sample-size framing (about 200 cans) for all sim metrics.
- That the reported dataset, ACT baseline, and tactile ablation come from the physical dataset,
  with the sim results marked smoke tests throughout.
