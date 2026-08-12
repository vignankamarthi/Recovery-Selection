# Related-work notes

Running staging file for the related-work sections of the two papers, the RA-L HARVEST dataset +
tactile ablation and the CoRL recovery-selection. Each entry is a citation plus how it positions
against our contributions, so the framing is settled before drafting.

## Tactile in vision-language-action (VLA) policies

### N0-VTLA (2026) -- the anchor

- **Cite.** NeoteAI Team and Fudan TEAI Team. "$N_0$-VTLA: Scaling Vision-Tactile-Language-Action Model
  with Latent Tactile Tokens." arXiv:2607.23782, 2026-07-26.
- **What it is.** A vision-tactile-language-action foundation model, claimed as the first VTLA pretrained
  on a large tactile corpus (their NeoData). Recipe, visuo-tactile pretraining, a staged predictive
  tactile pathway that predicts latent tactile tokens, and ALTER (advantage-conditioned offline RL) for
  offline policy improvement. Tactile is vision/image-based (contact read as an image on the fingers), not
  capacitive taxels.
- **Results.** It wins all nine real-robot NeoReal tasks, hits 63.8% mean success on a 20-task sim suite
  (vs 44.0% for the strongest baseline), and reaches 75-95% on three long-horizon real tasks with ALTER.
- **Why it matters to us.** It is independent, contemporaneous confirmation of the HARVEST premise. In
  contact-rich manipulation the decisive information at contact is force and slip, and vision-only policies
  fail there. It is the strongest available motivation for the tactile ablation.
- **How we position against it (it is motivation, not a competitor).**
  - Different contribution type. N0-VTLA is a policy architecture + a large-scale pretraining effort. Our
    RA-L contribution is a controlled, public, by-can dataset plus an isolated tactile-vs-vision ablation on
    a single deliberately slip-prone task, on a frozen ACT policy. The CoRL recovery-selection layer is
    orthogonal to it entirely.
  - We isolate the variable they conflate. A whole-system win (9/9 tasks) mixes architecture, tactile
    encoding, pretraining data, and task. HARVEST toggles exactly one thing, the tactile stream in vs out,
    on a fixed policy, across nominal vs damaged cans, reproducibly. That is the clean evidence the field is
    missing, not another system-level demo.
  - Different tactile modality. N0-VTLA uses vision/image-based tactile, while our TSF-85 is a capacitive
    taxel array (4x7) plus a dynamic slip channel and an IMU. Our ablation speaks to a different, cheaper,
    industrial sensor class.
  - Their "latent tactile tokens" are a modeling choice (a learned tactile encoding), which is downstream of
    our question. Our ablation asks whether tactile helps at all, upstream of how to encode it. The idea is
    a candidate later for the Part-2 competence signal, not for the ablation.

### The broader 2026 VTLA / tactile wave (adjacent, cite for context)

A cluster of 2026 papers puts tactile into manipulation policies. The RA-L related-work should therefore
show the thread is hot and then carve out our controlled-benchmark-plus-recovery angle. Seen while
sourcing N0-VTLA, confirm each before citing:
- OmniVTLA (arXiv:2508.08706), semantic-aligned tactile sensing in a VTLA.
- TacVLA (arXiv:2603.12665), contact-aware tactile fusion for robust VLA manipulation.
- AT-VLA (arXiv:2605.07308), adaptive tactile injection for feedback reaction in VLAs.
- UniTacVLA (arXiv:2606.31723), unified tactile understanding + prediction in VLAs.
- N0-TWAM (arXiv:2607.23783), a tactile-native world-action model (companion to N0-VTLA).

Takeaway, tactile-for-manipulation is a crowded 2026 frontier on the POLICY side. HARVEST's distinct value
is on the DATA + EVALUATION side (a controlled dataset and an isolated ablation) plus the recovery-selection
method, none of which these policy papers provide.
