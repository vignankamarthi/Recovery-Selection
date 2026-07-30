# Harvest-Recovery

A RIVeR Lab build that does two things in one repo, cleanly fenced. Read `ARCHITECTURE.md`
for how the software is built, and `PLAN.md` for what is done and what is next.

**Part 1, the HARVEST platform.** A multimodal data-collection framework for condition-aware robotic
manipulation of canned goods (seven synchronized streams, episode protocol, by-can dataset packaging),
built robot-free in simulation and ready to run on the real Kinova. On the collected dataset we train an
**ACT imitation-learning baseline** and run a **tactile ablation** (does tactile help the policy on
damaged cans). Targets **IEEE RA-L**. PI Prof. Taskin Padir, partner Good Shepherd Food Bank of Maine.

**Part 2, recovery-selection (our contribution).** Learning _which_ recovery a failed policy should run.
When a learned policy fails, the best recovery depends on the kind of failure, yet current systems
hard-code a single recovery behavior. We make recovery-strategy selection a learnable, cost-aware
decision (four competence tiers -> four recovery arms, chosen under a budget), and aim to show a small
learned selector beats any fixed single-mechanism baseline at matched intervention budget. The failures
the HARVEST policy makes on damaged cans are the substrate the recovery layer consumes. Targets **CoRL
2027** (ICRA workshop on-ramp).

## The task

Condition-aware in-hand pick-and-reorient. The robot grasps a can of unknown orientation and reorients it
in hand to expose the nutrition label, then the overhead camera verifies the label is legible and covers
enough of the frame. A dented or bulged can reorients differently than a nominal one, so it slips more,
and in-hand reorientation is exactly where tactile sensing earns its keep. The slip is a feature, it is
both the tactile signal Part 1 measures and the failure substrate Part 2 recovers from.

## Status

Software build phase complete, sim-first. Milestone 1 (the HARVEST framework + the ACT baseline pipeline)
and Milestone 2 (the full recovery layer + a sim dry-run of the make-or-break) are built and tested, and
the overhead label-visibility read is done, so all robot-free software is finished and the project is now
on the hardware on-ramp. The simulation is a **smoke test**, it validates the recording and training
pipeline but its ACT numbers are not trusted as evidence (see `ACT-EVAL-AUDIT.md` for why, the scripted
demo's target is not a learnable function of the observation). The reported ACT baseline, tactile
ablation, and the binding recovery make-or-break all come from the **hardware** dataset, where
demonstrations are human teleop. `proposal/PROPOSAL.tex` is the source of truth for scope and method.

## Quickstart

```bash
# tests (synthetic data only, macOS uses the CGL offscreen GL backend)
MUJOCO_GL=cgl .venv/bin/python -m pytest

# watch the simulator live (the scripted demonstration)
./watch_sim.sh              # optional [seed]

# generate the sim dataset locally (metadata by default; streams are heavy)
.venv/bin/python scripts/cluster/gen_dataset_parallel.py --help

# the AICR cluster run (materialize -> build LeRobot dataset -> train ACT -> eval).
# the playbook lives here.
cat scripts/cluster/README.md
```

## Layout

```
Harvest-Recovery/
├── README.md                       # this file: pitch, quickstart, the file map + doc index
├── ARCHITECTURE.md                 # how the software is built (the fence, data flow, seams, the sim)
├── HARVEST-VALIDITY-AUDIT.md       # red/blue validity audit of the sim dataset
├── ACT-EVAL-AUDIT.md               # red/blue audits of the ACT eval + the Option A decision
├── pyproject.toml                  # package config; pythonpath=["src"], pytest config
├── proposal/
│   ├── PROPOSAL.tex                # THE source of truth for scope + method
│   └── PROPOSAL.pdf                # compiled proposal (LaTeX build artifacts gitignored)
├── src/
│   ├── schema/                     # the CONTRACT: format-agnostic types, stdlib only (both tracks import this)
│   │   ├── episode.py              # ConditionClass, Outcome, CompetenceTier, Label, Episode, RecordedEpisode
│   │   └── streams.py              # Modality (the 7 streams), Sample, StreamSpec
│   ├── harvest/                    # PART 1: data framework + ACT baseline (imports schema, NEVER recovery)
│   │   ├── labels.py               # canonical label-name constants (shared producer <-> consumer)
│   │   ├── sensors/
│   │   │   ├── base.py             # SensorSource Protocol (the sensor seam)
│   │   │   └── mock.py             # MockSource: deterministic synthetic source for tests
│   │   ├── recorder/
│   │   │   └── recorder.py         # record_episode: sensor-agnostic sampling loop + timestamp check
│   │   ├── protocol/
│   │   │   └── protocol.py         # EpisodeProtocol FSM for a real collection session (hardware/mock path)
│   │   ├── control/                # backend-agnostic control layer (MuJoCo-free)
│   │   │   ├── backend.py          # RobotBackend + SceneOracle + GraspBackend Protocols (the hardware seam)
│   │   │   └── policy.py           # ManipulationPolicy Protocol + ScriptedGraspPolicy
│   │   ├── sim/                    # the MuJoCo world + demo generator (MuJoCo lives ONLY here)
│   │   │   ├── scene.py            # build the Gen3 + gripper + damaged-can MuJoCo model
│   │   │   ├── world.py            # SimWorld: thin control/read surface (delegates to ik/sensing/_render)
│   │   │   ├── ik.py               # damped-least-squares IK solvers (free functions)
│   │   │   ├── sensing.py          # force-torque / tactile / render / sample reads (free functions)
│   │   │   ├── _render.py          # the ONE process-global renderer, closed on model change (leak fix)
│   │   │   ├── reorient.py         # demonstrate(): grasp -> reorient -> present -> slip (THE demo generator)
│   │   │   └── episode.py          # record_sim_demo() + SimSource: run one demo, sample the 7 streams
│   │   ├── io/                     # serialization (on-disk formats live ONLY here)
│   │   │   ├── _serde.py           # episode <-> dict
│   │   │   ├── flat_npz_adapter.py # per-episode .npz + jsonl streams (the shipped sim path)
│   │   │   └── rosbag2_adapter.py  # rosbag2 read/write (the ROS2 / hardware boundary)
│   │   ├── dataset/
│   │   │   ├── splits.py           # by-can leak-free train/val/test (the can is the unit of independence)
│   │   │   ├── card.py             # caveat-forward dataset card
│   │   │   ├── export.py           # HuggingFace metadata rows
│   │   │   ├── competence.py       # proxy competence-tier tagging from a held-out margin
│   │   │   └── generate.py         # the grid driver (conditions x orientations x poses)
│   │   ├── policy/
│   │   │   └── trainer.py          # Trainer Protocol + StubTrainer (baseline floor) + LeRobotACTTrainer
│   │   ├── vision/
│   │   │   └── label_visibility.py # overhead label-visibility read (numpy-only; the SceneOracle's real-camera label read)
│   │   └── annotation/             # placeholder for hardware-phase labeling tools
│   └── recovery/                   # PART 2: recovery-selection (imports schema, NEVER harvest); built
│       ├── policy/base_policy.py   # BasePolicy Protocol + StubBasePolicy + FrozenACTPolicy (ACT stays frozen)
│       ├── competence/signals.py   # the 4 competence tiers + safe-set floor (Proxy/ACT competence models)
│       ├── failures/injection.py   # FailureMode catalog (5 kinds) + failure generators
│       ├── backend.py              # RecoveryBackend Protocol (the reset-and-replay seam)
│       ├── arms/                   # the four recovery arms (retry, rewind, replan, ask-human)
│       ├── grid/counterfactual.py  # counterfactual reset-and-replay grid + per-failure oracle
│       ├── metric/recovery_regret.py # ArmCost, CostWeights, recovery_regret + default weights
│       └── selector/selector.py    # cost-sensitive selector (Lagrangian human-budget)
├── scripts/cluster/                # the AICR run playbook + scripts (data is cluster-only)
│   ├── README.md                   # the cluster run playbook (order of operations)
│   ├── 00_setup_env.sh             # one-time env: Blackwell torch cu128 + lerobot + mujoco
│   ├── gen_dataset_parallel.py     # deterministic sharded episode generator (shard/assemble/local)
│   ├── 10_materialize.slurm        # regenerate the 600 streams to /work (persistent, no purge)
│   ├── 20_build_lerobot_dataset.py # build a LeRobotDataset from streams (--split train|heldout)
│   ├── 30_train_act.slurm          # lerobot-train the ACT policy on a B200
│   ├── 80_rigorous_eval.py         # held-out predicted-action L1 vs a no-move baseline (the valid eval)
│   └── 81_eval_diagnostic.py       # per-condition + sanity diagnostic
├── tests/                          # mirror src/, strict TDD, synthetic data only (run: MUJOCO_GL=cgl pytest)
│   ├── schema/test_schema.py           # the schema types
│   ├── harvest/test_mock_source.py     # MockSource
│   ├── harvest/test_recorder.py        # the recorder sampling loop
│   ├── harvest/test_protocol.py        # the EpisodeProtocol FSM
│   ├── harvest/test_pipeline.py        # end-to-end mock pipeline (record -> io -> read)
│   ├── harvest/test_io.py              # the io adapters round-trip
│   ├── harvest/test_splits.py          # by-can leak-free splits
│   ├── harvest/test_dataset.py         # dataset card + export
│   ├── harvest/test_competence.py      # competence-tier tagging
│   ├── harvest/test_generate.py        # the grid driver
│   ├── harvest/test_policy.py          # the Trainer interface + StubTrainer
│   ├── harvest/test_sim_world.py       # SimWorld physics/control/reads
│   ├── harvest/test_sim_condition.py   # per-condition can geometry
│   ├── harvest/test_sim_orientation.py # unknown-orientation grasp
│   ├── harvest/test_sim_episode.py     # record_sim_demo + SimSource
│   ├── harvest/test_reorient.py        # the demonstration generator
│   ├── harvest/test_label_visibility.py # the overhead label-visibility read (synthetic images + ROI)
│   └── recovery/                        # Part 2 suite: base_policy, competence, injection, backend, arms,
│                                        #   grid, metric, selector, smoke (+ the fenced sim_harness)
└── experiments/                    # committed eval result JSONs (the audit records)
```

The fence is enforced by import direction. `recovery` imports `schema` only, never `harvest`. `harvest`
never imports `recovery`. MuJoCo lives only in `harvest/sim/`, and the on-disk formats only in
`harvest/io/` (flat npz + jsonl for the sim path, rosbag2 for the ROS2/hardware boundary).
`ARCHITECTURE.md` explains all of this in depth.

## Documentation (the meta-doc standard list, kept current each milestone)

| Doc                         | What it is                                                               |
| --------------------------- | ------------------------------------------------------------------------ |
| `ARCHITECTURE.md`           | How the software is built (the fence, the data flow, the seams, the sim) |
| `proposal/PROPOSAL.tex`     | Source of truth for scope and method                                     |
| `HARVEST-VALIDITY-AUDIT.md` | Red/blue validity audit of the sim dataset                               |
| `ACT-EVAL-AUDIT.md`         | Red/blue audits of the ACT eval, and the Option A decision               |
| `scripts/cluster/README.md` | The AICR run playbook                                                    |

Internal working docs (gitignored, not in the public repo, kept locally as part of the meta-doc
standard): `PLAN.md` (live build plan), `CLAUDE.md` / `MEMORY.md` / `ANTIPATTERNS.md` (agentic project
files), `HARVEST-DESIGN-REVIEW.md` (Part 0 sign-off), `VENUE-TIMELINE.md` (venues and dates).

## Hard constraints

- Strict TDD, a failing test before implementation (synthetic data only in tests).
- By-can leak-free splits. The can is the unit of statistical independence, not the episode.
- The sim tactile ablation is a smoke test, never reported as evidence. The reported ablation is hardware.

## Hardware

Built robot-free so no time is lost while access is arranged. The target platform is a Kinova Gen3 +
Robotiq 2F-85 with TSF-85 tactile fingertips, a top-down overhead RGB-D camera, and the Gen3 wrist RGB-D,
on ROS2 Humble. Kinova POC is Drake Moore (coordinate via Teams). Compute runs on the AICR cluster via
`/Cluster-Compute`.
