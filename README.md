# Recovery-Selection

Learning *which* recovery a failed manipulation policy should run. When a learned
VLA policy fails, the best way to recover depends on what kind of failure occurred,
yet every current system hard-codes a single recovery behavior. This project makes
recovery-strategy *selection* a learnable, cost-aware decision problem, and aims to
show that a small learned selector chooses recoveries that beat any fixed
single-mechanism baseline at matched intervention budget. First-author robotics
work in the RIVeR Lab under Prof. Taskin Padir (robot + HRI coordinated with Lorena
Genua), targeting CoRL 2027.

## The gap

Robot-failure recovery is splintered into siloed lines that each commit to one
mechanism: retry/rewind (RaC, SPR), local-RL recovery (RecoveryChaining), VLM
re-planning (FailSafe, SC-VLA), and human hand-off (the Cornell HITL framework,
HRI 2026). Each works only in the regime it assumes. No published work (verified
over four substance-phrased literature probes during a structured search) does all
of: (a) formally defines a policy's *competence region* and types failures by their
relation to it; (b) learns a meta-policy selecting among three or more heterogeneous
recovery mechanisms under explicit cost/risk constraints; and (c) evaluates on real-
robot rollouts with monolithic end-to-end VLA policies.

## Contribution

- **Competence-grounded failure taxonomy.** Failure type is the state's relation to
  the frozen policy's competence region (latent-state density + action-head ensemble
  disagreement, with a control-invariant safe set as the hard anchor), not a semantic
  error label. The type implies the action.
- **A learned meta-policy over heterogeneous recoveries.** A cost-sensitive selector
  maps (failure features, competence signals, task context) to one of five recovery
  arms, with a Lagrangian budget on the ask-human arm.
- **Counterfactual recovery evaluation.** Reset-and-replay the full five-arm grid per
  injected failure to obtain a per-failure oracle and a new metric, *recovery-regret*
  (realized cost minus oracle cost), replacing binary recovery-success.

## Status

Proposal stage. Direction recommended by the topic-formation engine and confirmed by
an insurance probe; not yet formally locked. The PI/mentor-facing proposal is
`PROPOSAL.tex`. Build not started: the agentic 3-file standard and a TDD `src/`/`tests/`
skeleton will be added once the proposal is sent and the direction is locked.

## Planned layout

```
proposal/              contained LaTeX project: PROPOSAL.tex -> PROPOSAL.pdf    [here now]
src/                   competence model, failure injection, the grid, the selector  [planned]
  competence/          state-density + ensemble-disagreement calibration
  recovery/            the five recovery arms + the counterfactual grid runner
  selector/            cost-sensitive selector + recovery-regret metric
tests/                 mirror layout; TDD, synthetic data only                   [planned]
experiments/           per-run artifact dirs (sim pilot, UR3e validation)        [planned]
```

Reference (not in this repo): the understanding primer and the formation record live
in `../Topic Formation/`. Static identity `CLAUDE.md`, hard rules to come; gitignored.

## Method and plan

Simulation first, robot second. Weeks 1-4: build the competence model on a frozen VLA
(pi0-FAST/OpenVLA, inference or LoRA only), inject scripted failures, run the grid in
sim, fire the make-or-break gate (does the oracle beat the best fixed move by >15%
recovery-regret?). Weeks 4-10: train + calibrate the selector. Weeks 10-16: validate
on the UR3e (with Lorena), measure regret vs fixed baselines, ablate competence
signals. A NO-GO still ships a dataset + metric.

## Hard constraints

- Strict TDD: a failing test before any implementation (synthetic data only in tests).
- No subject/task leakage; sim and real reported separately; pre-registered falsifiers.
- No robot commands without Lorena; robot use coordinated through lab protocol.
- No commits or pushes from Claude (`/Commit-Initiation` plans only). No emojis.

## Hardware

NEU Explorer, 1x H200 (inference or LoRA only; no policy training). UR3e + RealSense
D455 + Azure Kinect, ROS2 Humble, for the Weeks 10-16 validation only.
