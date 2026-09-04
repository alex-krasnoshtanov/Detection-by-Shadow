# Experiments

A chronological record of what was tried in roughly twelve hours on
10 April 2026, with the numbers that were actually logged. Where a number was
not recorded, this says so rather than reconstructing it.

The final submission won the DEMCON Deep Tech track.

## Reading the numbers

Everything below is IoU, but measured on two different sets:

- **Leaderboard** — the organisers' score over the 414 test images, whose labels
  we never had. This is what the challenge was judged on and where 0.626 comes
  from.
- **Local validation** — mean IoU over a slice held out of the 1692 training
  frames, computed by our own code.

Same metric and the same scale, but different sets and different label
distributions, so read a leaderboard-against-validation comparison as
indicative rather than exact.

The v4 and v5 runs have **no local IoU at all**. By then the target was
decomposed and the training loss was a weighted sum over five outputs, so
validation tracked side and direction accuracy and never reassembled boxes to
score them. That is a genuine gap in the record: the v2-to-v4 improvement is
only visible on the leaderboard, and the local metric that made the v1/v2
diagnosis so clear was dropped exactly when it would have been most useful for
attributing the gain.

## The progression

| # | Approach | Validation | Leaderboard | Notebook |
| --- | --- | --- | --- | --- |
| baseline | Per-edge mean box | 0.4295 IoU | — | [01](../notebooks/01_eda.ipynb) |
| v0 | Classical shadow segmentation + optional SAM3, geometric heuristic | — | not submitted | [explorations/](../explorations) |
| v1 | ResNet-18, direct box regression, 320x320, GIoU | — | not recorded | [02](../notebooks/02_v1_direct_regression.ipynb) |
| v2 | ResNet-50, 576x384, GIoU + SmoothL1, freeze/unfreeze, TTA | **0.4675 IoU** | not recorded | [03](../notebooks/03_v2_giou_resnet50.ipynb) |
| v3 | Decomposed targets @ 384x384 (a teammate's baseline) | — | 0.590 | — |
| v4 | Decomposed @ native 720x480 + 19 shadow features + TTA | side 1.000, dir 0.547 | 0.614 | [04](../notebooks/04_v4_full_resolution.ipynb) |
| v5 | Decomposed @ 384x384, all 1692 samples, single seed | — | 0.604 / 0.618 / 0.620 | [05](../notebooks/05_v5_ensemble.ipynb) |
| v5-ens | Three seeds averaged + TTA | — | **0.626** | [05](../notebooks/05_v5_ensemble.ipynb) |

Per-seed leaderboard scores for v5: seed 777 → 0.604, seed 42 → 0.618,
seed 123 → 0.620. Ensembling the three → 0.626, i.e. **+0.006 over the best
individual seed** and +0.022 over the worst.

> The in-notebook `print` statements quote slightly different figures
> ("Their best: 0.590", "Our V4: 0.586"). Those strings were typed *before* the
> submissions were scored and are provisional; the table above uses the
> post-scoring annotations from the last cell of notebook 05, which are
> self-consistent across all five of our submissions.

## v0 — classical, then abandoned

The first hour went into a classical pipeline: shadow segmentation by
morphological and colour-space thresholding, connected-component analysis, then
a geometric heuristic projecting the shadow's principal axis to guess where the
caster stood. SAM3 was wired in behind a flag as optional mask refinement.

It was dropped without a submission. The heuristic never localised reliably
enough, and once the shadow's geometry was recognised as being useful *as
features for a learned model* rather than as a standalone solver, the CNN path
was clearly better value. The code is preserved in
[`explorations/`](../explorations) — it does not run, because a helper module it
imports was never committed.

The idea survived, though. The 19 descriptors in
[`features.py`](../src/shadow_detection/features.py) are exactly the quantities
this pipeline computed, repurposed as CNN inputs.

## v1 → v2 — direct regression, pushed to its limit

v1 regressed normalised `(xmin, ymin, xmax, ymax)` from a ResNet-18 under GIoU.
v2 added everything that ought to have helped:

- ResNet-50 (2048-dim features vs 512)
- aspect-preserving 576x384 instead of a squashed 320x320
- photometric augmentation (brightness, contrast, gamma, blur)
- backbone frozen for 8 epochs, then unfrozen at 0.1x learning rate
- a single cosine schedule instead of warm restarts, which had caused
  mid-training regressions
- GIoU + auxiliary SmoothL1, for gradients at zero overlap
- direction head on *detached* features, so it cannot corrupt box learning
- flip TTA
- automatic abstention on direction below 55% validation accuracy

Result at epoch 117 of 120, on 203 held-out images:

| Metric | Value |
| --- | --- |
| Mean IoU | 0.4675 |
| Median IoU | 0.4590 |
| IoU > 0.5 | 41.4% |
| IoU > 0.3 | 79.3% |
| IoU == 0 | 0.0% |
| Direction accuracy | 46.3% → abstained |
| Per-edge mean baseline | 0.4295 |

**+0.038 over predicting the average box.** Nothing was overlooked in the
engineering; the target parameterisation was wrong. Notably `IoU == 0` never
occurred — the model always found roughly the right region and never the right
box, which is the signature of a well-conditioned model on a badly-conditioned
target.

## v4 — the reparameterisation

Adopted the decomposed target formulation (`side` + four regressions) from the
teammate's baseline, added the 19 hand-crafted descriptors, and ran at
native 720x480 with no downscaling.

24,864,456 parameters. Batch 32, AdamW, head LR 1e-3 / trunk 1e-4,
ReduceLROnPlateau, 15-epoch patience. Early stopped at epoch 29 of 80 after
**7.9 minutes** on an RTX 6000 Ada.

| Metric | Value |
| --- | --- |
| Best validation loss | 1.3687 |
| Best validation side accuracy | **1.000** |
| Best validation direction accuracy | 0.547 |
| Leaderboard | 0.614 |

Two findings that shaped the rest of the day:

**Side classification is free.** 0.933 at epoch 0, 1.000 by epoch 1, and it
never dropped. Which side of the frame someone is on is trivially readable from
their shadow — so all remaining effort belonged on the distance regression.

**Direction is not learnable here.** It peaked at 0.547 against a 48.3% base
rate, and the confidence gate meant the v4 submission abstained on **all 414**
test images. Not a tuning failure; the signal is not in the data at this scale.
See [method.md](method.md#7-abstaining-on-direction).

Overfitting set in early — training loss fell from 1.07 at epoch 14 to 0.53 at
epoch 25 while validation loss rose from 1.37 to 2.16. With 1692 images and
25M parameters that is expected, and it is what made the next move obvious.

## v5 — all the data, three times

Two changes, both aimed at the 1692-sample ceiling rather than at the
architecture:

**Train on everything.** The 15% validation split was 254 images that could be
training data instead. Since v4 early-stopped at a consistent epoch and the
architecture was settled, the split had done its job. Trading it away means no
validation signal at all, so early stopping is replaced by a fixed 40-epoch
budget on a cosine schedule.

**Three seeds, averaged.** 42, 123, 777, averaged in decomposed space before
box reconstruction.

Dropping the input to a squashed 384x384 cut each run to ~9.6 minutes, which is
what made three of them affordable — about 29 minutes total for the ensemble.

| Submission | Leaderboard |
| --- | --- |
| seed 777 | 0.604 |
| seed 42 | 0.618 |
| seed 123 | 0.620 |
| **3-seed ensemble** | **0.626** |

Direction on the ensemble: 45 into frame, 93 out of frame, 276 abstained.

## The final hour — blending

Two artifacts, and the record is uneven:

**A weighted blend of our own five submissions** (`v5_ensemble` ×3,
`v5_seed123` ×2, `v5_seed42` ×1.5, `v4_fullres` ×1.5, `v5_seed777` ×1) ran
successfully and wrote `submission_MEGA.csv`. **Its score is not recorded.**

**A 0.7/0.3 blend** with a teammate's best submission
(leaderboard 0.601) is written out in the notebook but **the cell failed** with
`FileNotFoundError` — their CSV was not in the working directory. Whether the
blend was completed elsewhere is not recorded either. The failed cell is
preserved in [notebook 05](../notebooks/05_v5_ensemble.ipynb) as-is.

So **0.626 is the best score this repository can actually evidence.** The
blending machinery is reimplemented properly in
[`blend.py`](../src/shadow_detection/blend.py), with the id-alignment and
degenerate-box checks the hackathon version lacked.

## What did not work as expected

**Native resolution lost to a squashed 384x384.** v4 at a geometrically correct
720x480 scored 0.614; v5 at a distorted 1:1 384x384 scored 0.618–0.620 per
seed. Preserving the aspect ratio mattered less than the 3x speedup, which
bought all-data training and three seeds. Worth being precise about the
confound: v5 changed resolution *and* dropped the validation split, so this is
not a clean resolution comparison — the honest reading is that the extra data
and the ensemble outweighed the distortion, not that squashing helps.

**Ensembling gained +0.006, not the 2–5% expected.** The notebook's own header
predicted "2-5%" from three seeds; the actual gain over the best individual
seed was about 1%. Three seeds of the same architecture on the same data are
highly correlated, so there is not much variance left to average away.

**Direction was never learnable.** Covered above. Two architectures, two
regimes, both at chance.

**GIoU did not rescue a bad parameterisation.** Optimising the metric directly
is good advice that does not apply when the *coordinate system* is the problem.

## Reproducing

The dataset is not redistributed; see [dataset.md](dataset.md) for the layout.

```bash
# The best result: three seeds on all 1692 samples, ~29 min on one modern GPU
shadow-detection train \
  --train-dir data/train_data/train_data \
  --preset ensemble \
  --output-dir runs/v5-ensemble \
  --features-cache runs/features.npz

shadow-detection predict \
  --test-dir data/test_data/test_data \
  --sample-csv data/submission_example.csv \
  --checkpoints runs/v5-ensemble/model_seed*.pt \
  --output runs/submission_v5_ensemble.csv
```

```bash
# The v4 comparison: native resolution, held-out validation, early stopping
shadow-detection train \
  --train-dir data/train_data/train_data \
  --preset full-res \
  --output-dir runs/v4-full-res

shadow-detection predict \
  --test-dir data/test_data/test_data \
  --sample-csv data/submission_example.csv \
  --checkpoints runs/v4-full-res/model_seed42.pt \
  --native-resolution \
  --output runs/submission_v4.csv
```

### Batch size is the one setting you may have to change

The published runs used batch 128 on a 48 GB RTX 6000 Ada. Measured on a 12 GB
RTX 5070, one training step at 384x384 with AMP:

| Batch | Peak allocated | s/step | img/s |
| --- | --- | --- | --- |
| 32 | 4.45 GB | 0.134 | 238 |
| 48 | 6.70 GB | 0.219 | 220 |
| **64** | **8.78 GB** | **0.272** | **235** |
| 96 | 12.96 GB | 4.649 | 21 |
| 128 | 17.13 GB | 14.092 | 9 |

The cliff at 96 is not a gradual slowdown: it is the driver spilling to host
memory once the working set passes the 12.8 GB of VRAM, and it costs 11x. So on
a 12 GB card `--batch-size 64` is not a compromise, it is the setting -- it
holds full throughput at 8.8 GB with headroom. Anything larger is slower than
anything smaller.

Whole-epoch time on that card at batch 64 is about 20 s including the
dataloader, so the 40-epoch ensemble is roughly 13 minutes a seed.

Exact numbers will not reproduce bit-for-bit: cuDNN kernel selection and
`DataLoader` worker ordering are not pinned, so expect the same ballpark rather
than identical values. Each run directory carries its own `run.json` (full
config, per-epoch history, wall-clock) and `target_stats.json`, which is what
makes a checkpoint usable without the training data — something the original
notebooks did not save, and which cost real time.

## Checkpoints

My own v5 weights are gone: they lived on the university GPU server and were
not retrieved before access ended. The two surviving local checkpoints are from
the superseded v1 and v2 architectures and would not reproduce anything
documented here.

A trained model does survive, though. Filipp published one from the same
decomposed architecture as a TorchScript archive:

```bash
gh release download v1.0.0 --repo filipp-lotsmanov/shadow-detection
tar -xzf model_artifacts.tar.gz     # -> model.pt, target_stats.json
```

Its `target_stats.json` matches the values logged in
[notebook 04](../notebooks/04_v4_full_resolution.ipynb) to two decimal places
(208.58/80.85/172.90/309.33 against 208.59/80.87/172.91/309.34), which is a
useful independent check that both lineages standardised against the same data.

The v1 and v2 submission CSVs are in [`results/`](../results) as historical
artifacts.

## Cross-checking this rewrite against those weights

The rewrite in [`src/shadow_detection/`](../src/shadow_detection) reimplements a
feature extractor, a mirror map, a TTA scheme and a box reconstruction, any of
which could have drifted from the notebooks during the port. Running the
released weights through it is a direct check: a wrong feature ordering, a
broken mirror map or an inverted TTA reversal would all show up as collapsed
IoU, because the weights expect the original conventions exactly.

Over the eight frames Filipp publishes with ground truth:

| Metric | Value |
| --- | --- |
| Mean IoU | 0.782 |
| Median IoU | 0.812 |
| Range | 0.405 – 0.970 |
| IoU > 0.5 | 7 / 8 |
| Side classified correctly | 8 / 8, all at p = 1.000 |
| Direction | abstained on all 8 |

**This is not an evaluation.** All eight are training frames and the released
model trained on every one of them, so the number is optimistic by construction
and says nothing about generalisation. What it does establish is that this
package's inference path agrees with the code that produced the weights.

The rendered comparison is [`assets/predictions.png`](../assets/predictions.png).
Note that direction abstained on all eight even for a model whose author reports
65–70% direction accuracy at best — consistent with everything else here about
that head.
