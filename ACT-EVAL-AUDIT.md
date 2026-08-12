# ACT rigorous-eval red/blue audit (2026-07-24, post arm-slip fix)

Mandatory gate on the result-producing run `experiments/act_rigorous_eval.json`. RED and BLUE were
dispatched in parallel. This is the primary synthesis.

## What was claimed
"After the arm-slip fix made the demos teachable, the sim ACT baseline OVERFITS: it fits the 140
training cans (beats a no-move baseline ~8x) but does not generalize to the 60 held-out cans (worse
than no-move)."

## The numbers (validated as arithmetic)
- TRAIN: l1_eval 0.071 vs no-move 0.558 (beats ~8x); teacher-forced l1_loss 0.0057.
- HELD-OUT: l1_eval 0.772 vs no-move 0.600 (does NOT beat); teacher-forced l1_loss 0.050 (~9x train).
- Raw action range about +/-65 (NOT radians). chunk_size 100. action[t] = state[t+1].

## VERDICT: NO-GO on the "overfitting" claim. The sim gives software/pipeline validation ONLY, not an ACT ML-capability signal.
RED and BLUE are both right, and they are compatible. The measurement pipeline is now technically
correct (BLUE), but it measures a quantity that is unlearnable by construction (RED). The result
therefore reflects the demonstration generator, not ACT's capacity.

### What is solid (BLUE, and RED concurs on these)
1. The eval pipeline is faithful to LeRobot's own train/rollout usage. `make_pre_post_processors` +
   `pre(batch)` before `predict_action_chunk`, `post(pred)` after, `forward(pre(batch))` for the loss.
2. The normalization fix is proven. Train teacher-forced l1_loss 0.0057 matches the training-reported
   0.006, recovered by an independent code path. The earlier un-normalized version scored 1.12 on
   train (worse than no-move), which was the bug, now gone.
3. The by-can split is leak-free (each can_id in exactly one split). no-move is a legitimate trivial
   reference (action = state[t+1], baseline predicts state[t]). device/mask/denominator handling is
   correct. held-out is scored with the checkpoint's train-derived stats (correct), and its raw range
   sits inside the train range (no unnormalize clipping).
4. The arm-slip fix is genuinely in the arm: `_slip_roll` rolls the end-effector about the can's long
   axis via IK, all seven joints move, weld carries the can. The specific defect the PRIOR audit
   flagged (failure not in the action space) is fixed.

### FATAL to the interpretation (RED, and it holds up)
1. **The demo action is not a function of the observation, so held-out failure is guaranteed a priori.**
   Three hidden-state sources are welded into `action[t] = qpos[:7]`, none observable by ACT:
   - **Slip magnitude is a hash of the can id.** `slip_severity` adds `jitter = hash(seed)` in
     [-0.18, 0.18] of severity (~+/-9 deg of roll against a 23 deg cliff), seed = `can_seed_from_id`.
     ACT sees RGB + proprioception, never the seed, so the per-can slip is unpredictable on held-out.
   - **The presentation pose is a hidden per-can argmax.** `_search_present` sweeps 24 wrist rolls on
     a scratch sim and keeps the first clearing a pixel threshold (discontinuous, early-break). The
     glide interpolates the real arm to that snapshot. Which roll wins is a discretized function of
     geometry + IK null-space, not a smooth function of the two camera views.
   - **IK winding.** `move_pinch_pose` adds `dq` with no null-space regularization or joint wrapping,
     so continuous joints wind to tens of radians (the +/-65 range), path-dependent, not observable.
   Correct conclusion: the scripted sim demonstration is not a learnable function of the ACT
   observation, a property of the DEMONSTRATION GENERATOR, not evidence about ACT and not "overfitting."
   The arm-slip fix put the failure into the action space but NOT as a function of the observation, so
   the demos are still unteachable, in a new way.
2. **The +/-65 unit means the raw-space L1 is dominated by whichever single joint wound the most**, the
   most per-can-idiosyncratic, least observable quantity. Both the "8x beat" and the "worse than
   no-move" magnitudes are largely a statement about one unwrapped joint, not balanced arm-pose error.
   The script docstrings wrongly call the space "radians."

### Fixable weaknesses (independent of the fatal ones)
3. No-move over a 100-step chunk (episodes ~90 frames) is an inflated strawman, and train l1_eval is
   in-sample (measures convergence, not skill). A stronger baseline (nearest-neighbor, mean trajectory)
   was not tried.
4. The per-condition breakdown that would test the causal story (do nominal cans generalize, damaged
   not?) is unrunnable as built. `20_build_lerobot_dataset.py` writes a constant `task`, never the
   condition, so `81_eval_diagnostic.py` bucketed everything as "unknown."
5. Single seed, single run, no CI. Held-out is 60 per-can-deterministic cans, so 0.772 vs 0.600 (29%
   worse) is inside unquantified noise. Only the normalized teacher-forced 9x gap is a robust
   quantitative statement, and even it is a property of the demo generator.
6. `predict_action_chunk` (open-loop, z=prior) is not the deployed `select_action` (temporal ensemble)
   path. This is fine for an offline L1 but should be stated.

## What survives
- The pipeline runs end-to-end and the instrumentation is now trustworthy (software validation: PASS).
- The arm-slip fix is real and resolves the prior audit's defect.
- A train->held-out gap exists, but it is a property of the hidden-state-driven demo generator, not a
  lesson about ACT capacity or about generalization.

## Why this is not a project failure
It is consistent with the standing rule that the sim is a SMOKE TEST and the reported ACT baseline is
HARDWARE. Hardware demos are human teleop, which ARE a function of what the operator sees (no seed-hash
slip, no hidden argmax, consistent presentation strategy). The sim's job was to validate the recording
+ training pipeline, which it does. It cannot, by construction, provide an ACT generalization signal,
so the "weld-independent ML signal" sub-goal from the earlier reframe is not achievable in sim.

## OPTION B RE-RUN + AUDIT (2026-07-24, later). VERDICT: still NO-GO. The sim does not show ACT generalizing.
Option B was implemented (slip = pure function of the visible condition, continuous joints wrapped to
[-pi,pi], deterministic presentation search kept) and the full loop re-ran. Red + blue both validated.

RESULT (`experiments/act_rigorous_eval_optionb.json`), TRAIN l1_eval 0.045 vs no-move 0.408 (beats 9x),
HELD-OUT l1_eval 0.413 vs no-move 0.385 (does NOT beat, 7.2% worse). Pipeline still correct (train
teacher-forced l1 0.0062 == training loss 0.006).

The load-bearing finding (RED F1, BLUE concurs). The apparent "big improvement" (held-out went from 29%
worse than no-move to 7% worse) is MOSTLY A MEASUREMENT-UNITS ARTIFACT of the joint-wrapping. Wrapping
shrank BOTH ACT's L1 and the no-move baseline (no-move held-out 0.600 -> 0.385). The only cross-run
scale-invariant metric is the normalized teacher-forced held-out/train ratio, and it is UNCHANGED,
8.83x -> 8.47x (a 4% move). So the true, unit-free generalization gap did NOT narrow. Held-out ACT is a
statistical TIE with no-move on a single seed (n ~60 cans, not 22710 correlated frames), not resolvable
without >=3 seeds + a per-can bootstrap CI, but the pre-registered bar (held-out beats no-move) is NOT met.

Why (RED F4, structural, not fixable by training). Two of the three prior unlearnable factors are gone
(seed-hash slip, winding), but the third remains, the presentation is a discretized argmax over reachable
headings that a 96x96 RGB frame cannot resolve, so the observation UNDER-DETERMINES the action. Proof,
teacher-forced held-out l1 is still 8.5x train, meaning even handing the decoder the ground-truth action's
VAE latent, obs+latent cannot reconstruct held-out actions. A NEW artifact was also introduced. Wrapping
trades unbounded winding for a +/-pi seam (two near-identical poses across the seam differ by ~2pi in the
recorded action). A deeper tension also surfaced. A SIMPLER (learnable) presentation is not reachable
(a fixed heading reached ~2/10, and the search exists precisely because reachability is complex), so
reachability and learnability pull against each other in this sim.

HONEST RESTATEMENT of the result (RED's wording, adopted): "Option B removed two artifacts and produced a
small genuine improvement in the deployable eval ratio (ACT 29% -> 7% worse than no-move on held-out), but
the scale-invariant generalization gap is unchanged (~8.5x) and held-out ACT is a statistical tie with
no-move on one seed, so the sim still does not show ACT generalizing."

DECISION NEEDED (brought to Vignan). The sim has hit a STRUCTURAL wall, the scripted demo's presentation is
too complex for the observation, and making it simpler breaks reachability. Options:
- A (accept pipeline-validation-only). The story, pipeline works + ACT fits train + the sim cannot show
  generalization for a documented structural reason. Hardware teleop demos do not have this problem. This
  is the original Option A, which Vignan previously rejected, now re-surfaced with the evidence that Option
  B could not clear the bar.
- B2 (simplify the TASK, not just the code). Constrain spawn poses to a small canonical set so the
  presentation is one of a few simple, reachable, learnable classes, then re-gen/retrain. Real task change,
  uncertain payoff, moves further from the hardware task.
- Rigor add-ons (either way), >=3 seeds + per-can bootstrap CI + wire the per-condition breakdown into the
  dataset, to characterize the 7% and localize it (nominal vs damaged) rather than leave it single-seed.

## DECISION ARC (2026-07-24 -> 2026-07-25): B tried, then A chosen.
Vignan first chose Option B (make the demos learnable, re-run). B was implemented and re-audited (the
"OPTION B RE-RUN" section above). It removed the two fixable artifacts but the scale-invariant
generalization gap was unchanged (~8.5x) and held-out stayed a statistical tie with no-move, because the
remaining blocker (the presentation is a complex argmax the 96x96 observation cannot resolve, and a
simpler presentation is not reachable) is structural. **FINAL DECISION (2026-07-25): Vignan chose OPTION A,
accept the sim as pipeline-validation-only, the trusted ACT baseline is HARDWARE teleop (a function of what
the operator sees, so learnable by construction). Milestone 1 closes out as an honest smoke test.** The
original options as brought to Vignan the first time are kept below for the record.

## Decision options (as presented, per the gate)
- **A (recommended): accept the sim as software-validation-only.** Report honestly (pipeline validated,
  ACT trains + converges, arm-slip fix real, sim cannot give an ML-capability signal by construction).
  Do not rework the sim (respects "stop hammering this simulation"). Move to hardware for the real
  signal. This strengthens the RA-L honesty story rather than weakening it.
- **B: make the sim demos observation-learnable** (drop the seed-hash jitter so slip is a pure function
  of the observable condition; make the presentation a deterministic function of the observed can pose;
  wrap joints so no IK winding), then re-gen/retrain/re-eval. Substantial sim rework, contradicts the
  smoke-test stance.
- **C: keep A but add the cheap rigor RED asked for** (write condition into the dataset, per-condition
  breakdown; wrap/per-joint-normalize the L1; >=3 seeds + CI) purely to document the limitation
  defensibly, without trying to make the sim show learning.
