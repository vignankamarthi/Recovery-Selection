"""Canonical label names for the sim demonstration (Part 1).

Defined ONCE so the producer (`harvest.sim.episode.record_sim_demo`) and the consumers (the proxy
competence tagger `harvest.dataset.competence`) share the exact same strings. Before this, the
tier tagger looked labels up by string literal, so a rename in the producer would have silently
broken competence tagging. Importing these constants makes that a load-time error instead.

stdlib-only, harvest-internal (these labels are not part of the schema contract shared with
recovery; recovery reads only the `competence_tier` metadata key).
"""

from __future__ import annotations

# The two REAL graded stage signals plus the realized presentation margin.
UPRIGHT_SUCCESS = "upright_success"   # the reorient brought the label up to face the overhead camera
GRASP_STABLE = "grasp_stable"         # SIMULATOR default here (a real tactile/physics signal on hardware)
LABEL_VISIBLE = "label_visible"       # the overhead camera read enough label coverage
LABEL_UP_COS = "label_up_cos"         # the presented label's orientation cosine (held-out margin)
