# Detection by Shadow

**Locating a pedestrian who is entirely outside the camera frame, from the
shadow they cast into it.**

[![CI](https://github.com/alex-krasnoshtanov/Detection-by-Shadow/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-krasnoshtanov/Detection-by-Shadow/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Built for the DEMCON challenge at BrabantHack 2026. A vehicle-mounted camera
sees a shadow stretching in from the edge of frame; the person casting it is
off-screen, possibly about to step into the road. Predict their bounding box —
in coordinates *outside the image* — and whether they are walking into frame.

![Decomposing an off-frame bounding box](assets/decomposition.svg)

## Result

| | Leaderboard | Local validation |
| --- | --- | --- |
| Predict the average box per side | — | 0.4295 mean IoU |
| Direct box regression, fully tuned | not recorded | 0.4675 mean IoU |
| Decomposed targets, native resolution | 0.614 | side accuracy **1.000** |
| **Decomposed targets, 3-seed ensemble** | **0.626** | — |

The two columns are different metrics and are not comparable with each other;
[`docs/experiments.md`](docs/experiments.md) explains why, and records what was
and was not measured.

## The one idea that mattered

Every annotated person is *fully* outside the frame — `xmin` runs to −0.50 of
the image width, `xmax` to +1.44. Regressing those four corners directly asks a
network for coordinates on a scale nothing in its input anchors. Fully tuned —
ResNet-50, aspect-preserving input, GIoU loss, frozen-then-unfrozen trunk,
photometric augmentation, TTA — that approach reached 0.4675 mean IoU against a
0.4295 baseline of *predicting the average box*. All that machinery bought
0.038.

So describe the box relative to the edge it hides behind instead:

| Output | | |
| --- | --- | --- |
| `side` | classification | which edge — trivial, and it saturates at 100% |
| `distance_from_edge` | regression, px | the actual difficulty |
| `bbox_width`, `bbox_height` | regression, px | |
| `y_center` | regression, px | nearly constant (σ = 14 px), so nearly free |

Same backbone, same data, same loss family. This is what moved the project from
*marginally better than the mean* to a competitive score, and it pays off three
more times: the hard 1-D problem separates cleanly from the easy 1-bit one; all
four regression targets become invariant under horizontal flip, so mirror-TTA
needs no coordinate fix-up at all; and averaging an ensemble in this space
cannot produce a box belonging to neither member.

Nineteen hand-crafted shadow descriptors — where the dark region sits, how its
mass splits left/right, which way its principal axis leans, how deep it is
relative to the surrounding road — are fused onto the pooled trunk features
through a 512-unit bottleneck, supplying the spatial layout that global average
pooling throws away.

Full write-up: [`docs/method.md`](docs/method.md).

## Quickstart

```bash
git clone https://github.com/alex-krasnoshtanov/Detection-by-Shadow
cd Detection-by-Shadow
uv sync --extra dev          # or: pip install -e ".[dev]"
pytest                       # 155 tests, no dataset or GPU needed
```

The dataset is not redistributed — it belongs to the challenge organisers. See
[`docs/dataset.md`](docs/dataset.md) for the layout the loader expects. With it
in place, the best result reproduces in about half an hour on one modern GPU:

```bash
shadow-detection train \
  --train-dir data/train_data/train_data \
  --preset ensemble \
  --output-dir runs/v5-ensemble

shadow-detection predict \
  --test-dir data/test_data/test_data \
  --sample-csv data/submission_example.csv \
  --checkpoints runs/v5-ensemble/model_seed*.pt \
  --output runs/submission.csv
```

No dataset to hand? The descriptors run on any road photograph:

```bash
shadow-detection features path/to/frame.png --mirrored
```

`train` writes a self-contained run directory — weights, `target_stats.json`
(the standardisation constants inference needs) and `run.json` (full config,
per-epoch history, wall-clock). `predict` ensembles every checkpoint it is
given and validates the submission's id order before writing it. `blend`
weighted-averages finished submissions.

## Layout

```
src/shadow_detection/
  geometry.py    the reparameterisation: decompose / reconstruct / iou
  features.py    the 19 shadow descriptors, and their mirror map
  data.py        annotation loading, target standardisation, augmentation
  model.py       ResNet-50 trunk, geometry side-channel, three heads
  train.py       both regimes: held-out validation, or all-data + fixed budget
  predict.py     flip TTA, seed ensembling, submission guard rails
  blend.py       weighted blending of finished submissions
  cli.py         shadow-detection {train,predict,blend,features}

docs/            method, experiment log, dataset description
notebooks/       the five as-run hackathon notebooks, outputs preserved
explorations/    a classical + SAM3 pipeline, tried and dropped
results/         the submission CSVs that survive locally
tests/           155 tests, including a CPU train→predict→blend round trip
```

The notebooks are archives, not the interface — they carry the training logs
the numbers in the docs are quoted from, complete with server paths and a
preserved traceback. The package is the rewrite: the same method with the paths
as arguments, the mirror map in one place instead of three, and the
standardisation constants actually saved to disk.

## What this does not show

The results flatter the method, and it is worth saying how:

- **The frames are synthetic renders** — consistent lighting, clean shadows, no
  occlusion or clutter. Side classification hitting 100% says more about how
  legible a raytraced shadow is than about real dashcam footage.
- **1693 training images.** Enough to overfit a 25M-parameter model within
  25 epochs, and not enough to learn walking direction at all.
- **Direction was never learnable.** Two architectures, two training regimes,
  both at chance against a 48.3% base rate. The submissions abstain — the
  format accepts `-1`, and a coin flip is worse than declining.
- **The feature block was never ablated.** `ShadowNet(num_features=0)` builds
  the image-only comparison and is tested, but no run measured it. The v2→v4
  jump changed three things at once.
- **The best score is the best score that can be evidenced.** A cross-team
  blend was attempted in the final hour; that notebook cell failed and no
  result was recorded. 0.626 is what stands.
- **Feature thresholds were chosen by eye** and never swept.

[`docs/method.md#known-issues`](docs/method.md#known-issues) has the rest,
including one genuine bug — the principal-axis angle is not negated when
mirrored — kept because the published scores were produced with it, and pinned
by a test so it cannot be fixed silently.

## Credits

A four-person team split into two pairs for the twelve hours of the hackathon.

This repository is my own work — exploratory analysis, the v1/v2 direct-regression
line, the 19 shadow descriptors, the full-resolution and all-data ensemble runs,
and this rewrite. **The decomposed target formulation originated with the other
pair**, as their `384x384` baseline (leaderboard 0.590); adopting it is what
unlocked everything after v2, and the credit for the idea is theirs.
[Filipp Lotsmanov](https://github.com/FilippLotsmanov) was the other half of my
pair and contributed a parallel exploratory analysis, which is not included here.

Original hackathon repository (university account, full commit history):
[OleksiiKrasnoshtanov240247/Hackaton](https://github.com/OleksiiKrasnoshtanov240247/Hackaton).

## Licence

[MIT](LICENSE), matching the original repository. The challenge dataset is not
covered by it and is not included.
