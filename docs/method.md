# Method

The task: given a 720x480 frame containing a person's shadow but not the
person, predict the bounding box of the person — who is entirely outside the
frame — and whether they are walking into it.

![Decomposing an off-frame bounding box](../assets/decomposition.svg)

## 1. Reparameterising the target

The first two attempts regressed normalised `(xmin, ymin, xmax, ymax)`
directly, under a GIoU loss on the reasoning that the metric is IoU so IoU is
what should be optimised. That reasoning is fine. The target space is not.

Asking a network to emit `xmin = -0.42` means asking it for a coordinate
outside the image it is looking at, on a scale it has no anchor for. Nothing in
the frame is at `x = -0.42`. The four outputs are also strongly coupled — `xmin`
and `xmax` differ by a box width that barely varies — so the network spends
capacity learning that `xmax ≈ xmin + 0.11` instead of learning where the
person is. Pushed hard (ResNet-50, aspect-preserving input, photometric
augmentation, frozen-then-unfrozen trunk, TTA) this reached a validation mean
IoU of **0.4675** against a constant-prediction baseline of **0.4295**. A gain
of 0.038 for all of that machinery is a diagnosis, not a result.

The fix is to describe the box relative to the edge it hides behind:

| Output | Type | Why it is well-posed |
| --- | --- | --- |
| `side` | binary classification | Every person is *fully* off-frame, so `xmin < 0` identifies the side with no ambiguous middle case |
| `distance_from_edge` | regression, px | Always positive, mean 208.6 ± 45.1 — the actual difficulty of the task |
| `bbox_width` | regression, px | mean 80.9 ± 29.9 |
| `bbox_height` | regression, px | mean 172.9 ± 34.7 |
| `y_center` | regression, px | mean 309.3 ± 14.3 — nearly constant, so nearly free |

Reconstruction anchors on the box's *outer* corner and builds inwards:

```python
if side == LEFT:
    xmin = -distance_from_edge
    xmax = xmin + bbox_width
else:
    xmax = frame_width + distance_from_edge
    xmin = xmax - bbox_width
ymin, ymax = y_center - bbox_height / 2, y_center + bbox_height / 2
```

`decompose` and `reconstruct` are exact inverses, checked against real
annotation shapes in `tests/test_geometry.py`. The four regression targets are
standardised by [`TargetStats`](../src/shadow_detection/data.py) before the loss
sees them, because a single SmoothL1 over raw pixels would be dominated by
whichever target happened to be largest.

This change alone is what moved the project from "marginally better than
predicting the mean" to a competitive score. It also produces two free wins:

- **Side classification turns out to be trivial** — 100% validation accuracy
  within a couple of epochs. A shadow pointing right belongs to a person on the
  left. So the hard 1-D problem (where along the axis) is cleanly separated
  from the easy 1-bit problem (which axis end), and the model stops wasting
  capacity conflating them.
- **All four regression targets are invariant under horizontal flip**, because
  they are measured relative to whichever edge the person is behind rather than
  in absolute image coordinates. That makes flip-TTA and flip augmentation
  nearly trivial to implement correctly — see §4.

## 2. Nineteen hand-crafted shadow descriptors

A ResNet-50 pooled to a 2048-vector is good at "what kind of scene is this" and
oddly weak at "how far left does this shadow point". The global average pool is
partly to blame: it discards spatial layout, which is precisely the signal
here.

Rather than replace the pooling, the geometry is measured explicitly and fused
back in. [`features.py`](../src/shadow_detection/features.py) computes 19
descriptors in two groups:

**Row-median group (0–13).** A pixel is shadow if it is below 85% of the median
brightness of *its own row*. Per-row normalisation is the important detail: the
frames have a strong top-to-bottom brightness gradient (the far end of the road
is much brighter), so a global threshold either finds the whole near foreground
or nothing at all. From the resulting mask, restricted to the lower 60% of the
frame where the road is: centroid, spread, area fraction, the left/right mass
split, the peak and weighted-mean of the per-column density, the density in each
30-pixel margin, and the principal axis angle. Plus two probes of raw
brightness in the bottom-left and bottom-right 20-pixel columns, and their
ratio.

**Depth group (14–18).** Subtracting the image from a heavily blurred copy
(sigma = 20) gives a local-contrast map; values above 8 count as shadow. This
measures how *dark* the shadow is rather than where it is, which separates a
crisp nearby shadow from a faint distant one — a distance cue the positional
features cannot express.

The 19-vector is concatenated onto the pooled 2048 and the sum is projected
through a single `Linear(2067 → 512)` bottleneck before the heads. The
bottleneck is what makes the descriptors matter at all: concatenated straight
onto a 2048-vector feeding a linear head, 19 numbers would be swamped. Forcing
everything through a narrow layer makes the network spend capacity on them.

`ShadowNet(num_features=0)` builds the image-only ablation, which is the honest
way to measure what the block actually buys. That ablation was never run — see
[Known issues](#known-issues).

## 3. Architecture

```
image ──> ResNet-50 (ImageNet init) ──> avgpool ──> 2048
                                                     │
19 shadow descriptors ───────────────────────────────┼──> concat (2067)
                                                     │
                                    Linear 512 + BN + ReLU + Dropout(0.3)
                                                     │
                  ┌──────────────────────────────────┼──────────────────┐
              side (2)                       regression (4)      direction (2)
```

24,864,456 parameters, pinned by a test so an accidental architecture change
cannot pass while the docs still claim the published result.

One trunk and three heads rather than three models: the predictions share
almost all of their evidence — where the shadow points gives the side, its
length and faintness give the distance, the shape of its near end carries
whatever direction signal exists. The loss is
`CE(side) + 5·SmoothL1(regression) + CE(direction)`. The 5x is not tuning
folklore: SmoothL1 on standardised targets is numerically much smaller than two
cross-entropy terms, and without it the regression signal is drowned out.

The trunk keeps ResNet's global average pool, so any input resolution collapses
to 2048 features. That is what let the same architecture run unchanged at both
384x384 and the native 720x480. Parameter groups give the ImageNet-initialised
trunk a 10x lower learning rate than the heads, which start from noise — a
single rate either crawls or destroys the pretrained features in the first few
batches.

## 4. Flip augmentation, done consistently

Horizontal flip is the only geometric augmentation this problem admits, and it
is a genuine doubling of the data because the left/right split is 852/840. But
mirroring the image changes three things at once:

1. `side` inverts — a person behind the left edge is now behind the right.
2. `direction` inverts — walking into frame from the left becomes walking into
   frame from the right, which flips the sign convention.
3. The **hand-crafted features must be mirrored too.**

Miss the third and the network is shown a left-leaning shadow labelled "person
on the right" half the time. That is worse than not augmenting, and it produces
no error, no warning, and a quietly worse model. The notebooks open-coded the
index juggling in three separate places; here it is one function,
`mirror_features`, with the map declared as data:

- `LEFT_EDGE_BRIGHTNESS` ↔ `RIGHT_EDGE_BRIGHTNESS`, and
  `LEFT_MARGIN_DENSITY` ↔ `RIGHT_MARGIN_DENSITY` swap.
- `EDGE_BRIGHTNESS_RATIO` is recomputed from the swapped pair, so it inverts.
- `SHADOW_CENTROID_X`, `SHADOW_MASS_LEFT_FRACTION`, `SHADOW_PEAK_COLUMN` and
  `SHADOW_WEIGHTED_COLUMN` map to `1 - value`.
- Everything else describes size, depth or vertical position, and is untouched.

`tests/test_features.py` asserts the property directly — features computed on a
genuinely flipped image equal `mirror_features` applied to the original — for
every index, including the ones that must *not* change.

## 5. Test-time augmentation

Because the four regression targets are flip-invariant (§1), TTA is unusually
clean: run the image and its mirror, **average the four regressed values
directly**, and reverse the class order of the two classifier outputs before
averaging them.

```python
regression = (regression_original + regression_flipped) / 2       # no fix-up
side_probs = (side_original + side_flipped[::-1]) / 2             # un-flip
```

The reversal is easy to omit and hard to notice: averaging the raw softmax
vectors pulls every prediction towards 50/50, which looks like an
under-confident model rather than a bug. `tests/test_predict.py` reconstructs
the averaging by hand and asserts that the naive version is measurably
different.

Compare with the earlier direct-regression version, which had to map
`xmin ← 1 - xmax_flipped` and got the index bookkeeping wrong twice before it
was right.

## 6. Ensembling

Once a validation split was no longer being held out (see
[experiments.md](experiments.md)), three models were trained from seeds
42, 123 and 777 on all 1692 samples and averaged. Averaging happens in
*decomposed space, before reconstruction*, which matters: if the ensemble
averaged reconstructed boxes and two members disagreed about the side, the
result would land in the middle of the frame — where no annotated person ever
is — overlapping neither answer. Averaging the side probabilities and taking a
single argmax means the ensemble always commits to one side.

The same caveat applies to blending finished CSVs, which is all
[`blend.py`](../src/shadow_detection/blend.py) can do when the only thing shared
between two models is their submission file. It warns when a blended box is not
fully outside the frame, because that is the signature of exactly this failure.

## 7. Abstaining on direction

`walking_into_frame_bool` was the part of the task the shadow simply did not
carry. Validation direction accuracy across every run:

| Run | Best validation direction accuracy |
| --- | --- |
| v2 (direct regression) | 0.463 |
| v4 (decomposed, full-res) | 0.547 |

Against a 48.3% base rate, both are chance. The shadow tells you where someone
is standing; it does not reliably tell you which way they are facing, at least
not at 1692 samples.

The submission format accepts `-1` for "no prediction", so the right move is to
decline. The models emit a direction only when the ensemble's softmax exceeds
0.6, and abstain otherwise:

- v4 abstained on **all 414** test images — its confidence never once cleared
  the threshold.
- v5's ensemble committed on 138 and abstained on 276.
- The final blended submissions set `direction = -1` everywhere.

Guessing on a coin-flip signal costs accuracy and buys nothing. Reporting
"chance, so we declined" is the honest version of a negative result, and it is
cheaper than a confidently wrong answer.

## Known issues

Kept as they are, because the published leaderboard scores were produced with
them and reproducibility beats tidiness. They are documented rather than
silently fixed.

**The principal-axis angle is not mirrored.**
`Feature.SHADOW_PRINCIPAL_ANGLE` passes through `mirror_features` unchanged,
but a mirrored axis should negate its angle. So on flipped samples this one
feature is wrong. Its likely impact is small — it is 1 of 19 inputs, it is
undefined for weak shadows (fewer than 100 mask pixels leaves it at 0), and the
positional features carry the same orientation information more robustly — but
"likely small" is not "measured". `tests/test_features.py` asserts the
discrepancy exists, so fixing it fails a test that points back at this entry.

**`SHADOW_PEAK_COLUMN` mirrors only up to `argmax` tie-breaking.** `argmax`
returns the first maximal index, so on a shadow with a flat column-density
plateau the original picks its left end and the mirror picks its right end.
Real shadows taper, so this is a corner case; `tests/test_features.py` pins the
exact behaviour.

**The feature block was never ablated.** `ShadowNet(num_features=0)` exists and
is tested, but no run compared it against the full model. The claim that the 19
descriptors help rests on the jump from v2 to v4, which changed the target
parameterisation *and* the resolution *and* added the features — three
confounded variables. Honest summary: the reparameterisation is clearly
responsible for most of the gain, and the features' individual contribution is
unmeasured.

**Feature thresholds are unjustified.** The 0.85 row-median ratio, the 0.40 and
0.58 region cutoffs, the sigma of 20, the contrast threshold of 8 — all were
chosen by eye during a hackathon and none were swept. They are named constants
in `features.py` rather than magic numbers, which makes a sweep easy, but the
sweep did not happen.

**The 384x384 input squashes a 3:2 frame to 1:1.** This is not a bug, and it is
what the best run did; see [experiments.md](experiments.md#what-did-not-work-as-expected)
for why the distorted-but-faster input beat the geometrically correct one.
