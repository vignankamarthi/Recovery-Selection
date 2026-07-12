# Harvest-Recovery

A RIVeR Lab build that does two things in one repo, cleanly fenced. **Part 1** is the
HARVEST data-collection framework (multimodal robot manipulation of canned goods:
synchronized recording, episode protocol, annotation, dataset packaging), built
robot-free in simulation and ready to run on the real Kinova. **Part 2**, our research
contribution, is *recovery-selection*, learning *which* recovery a failed manipulation
policy should run.

When a learned VLA policy fails, the best way to recover depends on what kind of
failure occurred, yet every current system hard-codes a single recovery behavior.
Recovery-selection makes recovery-strategy *selection* a learnable, cost-aware decision
problem, and aims to show that a small learned selector beats any fixed single-mechanism
baseline at matched intervention budget. The failures the HARVEST policy makes on
damaged cans are not noise to discard; they are the substrate the recovery layer
consumes. First-author robotics work under Prof. Taskin Padir (robot + HRI coordinated
with Lorena Genua), targeting CoRL 2027.

## The gap

Robot-failure recovery is splintered into siloed lines that each commit to one
mechanism: retry/rewind (RaC, SPR), local-RL recovery (RecoveryChaining), VLM
re-planning (FailSafe, SC-VLA), and human hand-off (the Cornell HITL framework,
HRI 2026). Each works only in the regime it assumes. To our knowledge, no single
system does all of: (a) types failures by their relation to a policy's *competence
region*; (b) learns a meta-policy selecting among three or more heterogeneous recovery
mechanisms (not binary act-versus-ask or act-think-abstain routing) under explicit
cost/risk constraints; and (c) evaluates on real-robot rollouts with monolithic
end-to-end policies.

## Contribution

- **Competence-grounded failure taxonomy.** Failure type is the state's relation to
  the frozen policy's competence region (latent-state density + action-head ensemble
  disagreement, with a control-invariant safe set as the hard anchor), not a semantic
  error label. The type implies the action.
- **A learned meta-policy over heterogeneous recoveries.** A cost-sensitive selector
  maps (failure features, competence signals, task context) to one of four recovery
  arms (retry, rewind and re-approach, re-plan, ask-human), with a Lagrangian budget on
  the ask-human arm, which is also the terminal safety fallback. The control-invariant
  safe set is a hard floor beneath the arms, not a selectable arm.
- **Counterfactual recovery evaluation.** Reset-and-replay the full four-arm grid per
  injected failure to obtain a per-failure oracle and a new metric, *recovery-regret*
  (realized cost minus oracle cost), replacing binary recovery-success.

## Status

Greenlit by Padir 2026-06-19; proposal finalized and sent 2026-07-06; software build
phase. `proposal/PROPOSAL.tex` is the source of truth. Part 0 (HARVEST design review)
is done and folded into the proposal. Now cranking through all software robot-free
(Phase 1 framework, then Phase 2 recovery) before touching any hardware, so collection
can begin the day the arm is available. The internal plan is `PLAN.md`; the strategy
record is `../HARVEST-TOUCH-REVIEW.md`.

## Layout

```
proposal/              LaTeX: PROPOSAL.tex -> PROPOSAL.pdf  (augmented proposal lands here)
src/
  schema/              the contract: episode + stream representation (dependency-light)
  harvest/             PART 1: sensors/ recorder/ protocol/ sim/ io/ annotation/ dataset/
  recovery/            PART 2: competence/ arms/ grid/ metric/ selector/   (on top, fenced)
tests/                 mirrors src/; TDD, synthetic data only
experiments/           per-run artifact dirs (planned)
```

The fence (enforced by import direction): `recovery` imports `schema` only, never
`harvest` internals. `harvest` never imports `recovery`. The on-disk format (rosbag2)
is known only inside `harvest/io/`. Reference (not in this repo): the understanding
primer and the formation record live in `../Topic Formation/`.

## Method and plan

Part 1 first, robot-free: build the HARVEST data-collection framework against physics-sim
episodes (MuJoCo/robosuite), recording all streams time-aligned into rosbag2 (via the
pure-Python `rosbags` library, no ROS2 on macOS), with by-can splits and annotation.
Then Part 2 on top: build the competence model on a frozen policy (inference or LoRA
only), inject scripted failures, run the four-arm counterfactual grid, and fire the
make-or-break gate (does the per-failure oracle beat the best fixed move by >15%
recovery-regret, and is that headroom learnable from the competence signals?). Real-robot
validation follows once hardware is arranged. A NO-GO still ships a benchmark + the
recovery-regret metric.

## Hard constraints

- Strict TDD: a failing test before any implementation (synthetic data only in tests).
- No subject/task leakage; sim and real reported separately; pre-registered falsifiers.
- No robot commands without Lorena; robot use coordinated through lab protocol.
- No commits or pushes from Claude (`/Commit-Initiation` plans only). No emojis.

## Hardware

The framework is built robot-free so no time is lost while hardware access is arranged;
physical collection on the real Kinova is the primary deliverable once the arm is
available. HARVEST target platform: Kinova Gen3 + Robotiq 2F-85 gripper with TSF-85
tactile fingertips + a top-down overhead RGB-D camera (Azure Kinect) + the Gen3 wrist
RGB-D module, ROS2 Humble (coordination contact PENDING). Compute: the AICR cluster
(Massachusetts AI Compute Resource) via `/Cluster-Compute`, a single strong GPU (B200/RTX),
inference or LoRA only, no policy training, for the Part 2 competence/selector work.
