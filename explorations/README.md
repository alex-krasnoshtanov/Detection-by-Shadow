# Explorations

A direction that was tried and dropped, kept for provenance.

## `classical_sam_pipeline.py`

The first attempt at the challenge, before any neural network was trained:
segment the shadow, describe its geometry, and project backwards to guess where
the person casting it must be standing.

- **Shadow segmentation** by classical means — colour-space and morphological
  thresholding with connected-component filtering (`classical_shadow_mask`).
- **Per-component geometry** — area, elongation, principal axis, the position
  of the component's near end (`describe_components`).
- **Box prediction** by projecting the primary component's principal axis out
  of the frame and estimating a person-sized box at the far end
  (`predict_shadow_target`).
- **SAM3 refinement**, optional and off by default, as a better mask source
  than thresholding (`build_sam_shadow_mask`), with the classical mask as a
  fallback.
- An ISTD-dataset evaluation mode, for checking the segmentation against a
  public shadow-detection benchmark.

### Why it was dropped

It never localised reliably enough to be worth a submission. The projection
step assumes the shadow's principal axis points at the caster, which holds for
a long clean shadow on flat ground and degrades badly otherwise. Once it became
clear that these geometric measurements were valuable as *features for a
learned model* rather than as a standalone solver, the CNN path was obviously
better value for the remaining hours.

The idea did survive: the 19 descriptors in
[`features.py`](../src/shadow_detection/features.py) are essentially the
quantities this pipeline computed, repurposed as inputs to the network. That
turned out to be the right home for them.

### It does not run

This file is preserved verbatim and is **not executable**. It imports a helper
module, `istd_dataset`, that was never committed to the original repository and
no longer exists. It also expects `opencv-python` and `transformers`, neither of
which is a dependency of this project.

It is excluded from linting and from the test suite, and nothing in
`src/shadow_detection/` imports it. Read it as a record of an approach, not as
code to use.
