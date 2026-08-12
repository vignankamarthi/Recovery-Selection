# Real can-label reference photos

Photos of a real Campbell's condensed tomato soup can (10.75 oz), taken 2026-07-30. These are the
specimens that validated the overhead label read for hardware:

- `campbell_tomato_facts.png` / `campbell_tomato_facts_2.png` -- the Nutrition Facts panel (white text on
  the saturated red brand band).
- `campbell_tomato_side.png` -- a side view of the label wrap.

## What they showed

Running `harvest/vision/label_visibility.py` on these frames proved the sim's near-white default spec does
NOT transfer to a real can, since it matched only ~0.2% of the label. An HSV band on the Campbell brand red
matched cleanly (~6%) and stayed off the bare steel ends, a light arm, and the table. That result is the
`CAMPBELL_RED_SPEC` preset in `label_visibility.py`. Retune the HSV bands under the real overhead camera
and lighting before trusting the coverage numbers.

Full-resolution phone photos (~8 MB each). If the repo should stay light, downsample or gitignore them.
They are a reference, not a pipeline input.
