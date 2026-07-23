# Harvest-Recovery

A RIVeR Lab build that does two things in one repo, cleanly fenced.

**Part 1, the HARVEST platform.** A multimodal data-collection framework for condition-aware robotic manipulation of canned goods (synchronized recording, episode protocol, annotation, by-can dataset packaging), built robot-free in simulation and ready to run on the real Kinova. On the collected dataset we train an **ACT imitation-learning baseline** and run a **tactile ablation** (does tactile help the policy on damaged cans). This is the HARVEST-Touch dataset + baseline, targeting **IEEE RA-L**. PI Prof. Taskin Padir, partner Good Shepherd Food Bank of Maine, robot + HRI coordinated with Lorena Genua.

**Part 2, our research contribution: recovery-selection.** Learning *which* recovery a failed manipulation policy should run. When a learned policy fails, the best way to recover depends on what kind of failure occurred, yet every current system hard-codes a single recovery behavior. Recovery-selection makes recovery-strategy *selection* a learnable, cost-aware decision problem, and aims to show that a small learned selector beats any fixed single-mechanism baseline at matched intervention budget. The failures the HARVEST policy makes on damaged cans are not noise to discard, they are the substrate the recovery layer consumes. Targets **CoRL 2027** (ICRA workshop on-ramp).

## The task

Condition-aware in-hand pick-and-reorient. The robot grasps a can of unknown orientation and reorients it in hand to expose the nutrition label, then a camera verifies the label is legible and covers enough of the frame. A dented or bulged can has different grasp affordances and in-hand reorientation dynamics than a nominal one, so failures are frequent, varied, and safety-relevant, and in-hand reorientation is exactly where tactile sensing helps most.

## The gap (Part 2)

Robot-failure recovery is splintered into siloed lines that each commit to one mechanism, whether retry/rewind (RaC, SPR), local-RL recovery (RecoveryChaining), VLM re-planning (FailSafe, SC-VLA), or human hand-off (the Cornell HITL framework, HRI 2026). Each works only in the regime it assumes. To our knowledge, no single system does all three at once, (a) typing failures by their relation to a policy's *competence region*, (b) learning a meta-policy that selects among three or more heterogeneous recovery mechanisms under explicit cost/risk constraints, and (c) evaluating on real-robot rollouts with monolithic end-to-end policies.

## Contribution (Part 2)

- **Competence-grounded failure taxonomy.** Failure type is the state's relation to the frozen ACT policy's competence region (latent-state density + action-head ensemble disagreement, with a control-invariant safe set as the hard anchor), not a semantic error label. The state's position sorts it into one of four competence tiers.
- **A learned meta-policy over heterogeneous recoveries.** A cost-sensitive selector maps (failure features, competence signals, task context) to one of four recovery arms (retry, rewind and re-approach, re-plan, ask-human), under a Lagrangian budget on the ask-human arm. The tier is the competence prior, and the selector is the cost-aware chooser that can deviate from the default arm under budget and risk. The control-invariant safe set is a hard floor beneath the arms, not a selectable arm.
- **Counterfactual recovery evaluation.** Reset-and-replay the full four-arm grid per injected failure to obtain a per-failure oracle and a new metric, *recovery-regret* (realized cost minus oracle cost), replacing binary recovery-success.

## Status

Software build phase, sim-first. `proposal/PROPOSAL.tex` is the source of truth. Padir's original brief (`../HARVEST Touch Intern Project.docx`) is the draft it corrects and extends. Part 0 (HARVEST design review, flags F1-F7) is done and folded into the proposal. We build all software robot-free (Part 1 framework + ACT baseline, then Part 2 recovery) before touching hardware, so collection begins the day the arm is available. The internal plan is `PLAN.md`.

## Layout

```
proposal/              LaTeX: PROPOSAL.tex -> PROPOSAL.pdf
src/
  schema/              the contract: episode + stream representation (dependency-light)
  harvest/             PART 1: sensors/ recorder/ protocol/ sim/ io/ annotation/ dataset/ policy/
  recovery/            PART 2: competence/ arms/ grid/ metric/ selector/   (on top, fenced)
tests/                 mirrors src/; TDD, synthetic data only
experiments/           per-run artifact dirs (planned)
```

The fence (enforced by import direction): `recovery` imports `schema` only, never `harvest` internals. `harvest` never imports `recovery`. The on-disk format (rosbag2) is known only inside `harvest/io/`.

## Method and plan

Part 1 first, robot-free. Build the HARVEST framework against MuJoCo physics-sim episodes, recording all streams time-aligned into rosbag2 (via the pure-Python `rosbags` library, no ROS2 on macOS), with by-can leak-free splits and annotation. Train the ACT baseline on the sim dataset and run the tactile ablation as a pipeline smoke test (simulated tactile is synthetic, so the reported ablation result comes from the physical data). Then Part 2 on top. Build the competence model on the frozen ACT, inject scripted failures, run the four-arm counterfactual grid, and fire the make-or-break gate (does the per-failure oracle beat the best fixed move by >15% recovery-regret, and is that headroom learnable from the real ACT competence signals). A NO-GO still ships a benchmark + the recovery-regret metric. Real-robot collection and the reported tactile ablation follow once hardware is arranged.

## Hard constraints

- Strict TDD, a failing test before any implementation (synthetic data only in tests).
- By-can leak-free splits. The can is the unit of statistical independence, not the episode.
- Grasp-stability labels from simulator ground truth in sim, from an independent source (overhead-vision success or a human) on the real robot, never from the tactile stream.
- The sim tactile ablation is a smoke test, never reported as evidence.
- No robot commands without Lorena. No commits or pushes from Claude (`/Commit-Initiation` plans only). No emojis.

## Hardware

The framework is built robot-free so no time is lost while hardware access is arranged. Physical collection on the real Kinova is the primary deliverable once the arm is available. The target platform is a Kinova Gen3 + Robotiq 2F-85 gripper with TSF-85 tactile fingertips + a top-down overhead RGB-D camera (Azure Kinect) + the Gen3 wrist RGB-D module, ROS2 Humble (coordination contact and TSF-85 mounting PENDING, confirmed when Padir is back). Compute runs on the AICR cluster (Massachusetts AI Compute Resource) via `/Cluster-Compute` for ACT training and the recovery selector, with a lab GPU as the hardware-time fallback.
