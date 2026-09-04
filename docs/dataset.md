# The dataset

The data comes from the DEMCON challenge at BrabantHack 2026 and is **not
redistributed here**. It is not ours to license, and the repository's
`.gitignore` excludes `data/` so it cannot be committed by accident. The only
frames that appear are illustrative: a handful inside the EDA notebook's saved
plots, and the eight rendered into
[`assets/predictions.png`](../assets/predictions.png).

Everything below was measured on the training split during the hackathon; the
numbers are reproduced in [`01_eda.ipynb`](../notebooks/01_eda.ipynb).

## Expected layout

```
data/
├── train_data/train_data/
│   ├── image_0.png          720 x 480, RGB
│   ├── image_0.json         annotation, stem matches the image
│   └── ...                  1693 pairs
├── test_data/test_data/
│   ├── image_1.png
│   └── ...                  414 images, no annotations
└── submission_example.csv   defines the required id order
```

`build_samples` globs recursively and pairs each annotation with the image
whose stem matches the annotation's own `file_name` field, so the exact nesting
does not matter as long as the pairs are somewhere under the directory you pass
to `--train-dir`.

## Annotation format

```json
{
  "file_name": "image_0",
  "walking_into_frame_bool": 1,
  "bbox": {
    "top_left":     [-228.87714385986328, 223.6187400817871],
    "top_right":    [-110.61487913131714, 223.6187400817871],
    "bottom_left":  [-228.87714385986328, 416.3587894439697],
    "bottom_right": [-110.61487913131714, 416.3587894439697]
  }
}
```

All four corners are stored even though every box is axis-aligned (checked
across all 1693 annotations), so half the coordinates are redundant.
`load_annotation` collapses them by taking extremes rather than trusting
`top_left` to be the minimum corner — cheap insurance against a corner-order
surprise silently producing a negative-width box.

Note the negative x coordinates. That is the whole problem.

## What the numbers say

| Property | Value |
| --- | --- |
| Frames | 1693 train, 414 test |
| Resolution | 720 x 480, constant |
| Boxes axis-aligned | all of them |
| Person behind the left edge | 852 (50.4%) |
| Person behind the right edge | 840 (49.6%) |
| Walking into frame | 48.3% |
| Mean box size, normalised | 0.112 W x 0.360 H |
| Mean aspect ratio (h/w) | 2.4 |

Four findings drove every later decision:

**The person is always completely outside the frame.** Not clipped by the edge
— fully beyond it. `xmin` normalised spans `[-0.499, 1.296]` and `xmax` spans
`[-0.351, 1.443]`; there is no sample where the box overlaps the image at all.
This is what makes the reparameterisation in
[`geometry.py`](../src/shadow_detection/geometry.py) safe: the sign of `xmin`
alone identifies the side, with no ambiguous middle case.

**Left and right are almost exactly balanced.** 852 against 840. So horizontal
flip is a *free* doubling of the data — it maps a valid left sample onto a
valid right sample without distorting the class balance.

**The vertical axis is nearly constant.** `ymin` normalised lives in
`[0.441, 0.480]` and `ymax` in `[0.720, 0.968]`; in pixels, `y_center` has a
standard deviation of just 14.3 against a mean of 309.3. The camera height and
pitch are fixed and people are roughly the same size, so vertical placement is
almost fully determined. Predicting the training mean for `y_center` is already
close to optimal, and the real difficulty is entirely horizontal: *how far past
the edge*.

**The frames are synthetic renders.** Consistent lighting, clean shadows, no
occlusion or clutter. This is worth stating plainly because it flatters the
results: side classification reaching 100% validation accuracy
([`docs/experiments.md`](experiments.md)) says more about how legible a
raytraced shadow is than about how the method would fare on real dashcam
footage. See "What this does not show" in the README.

## Baselines to beat

| Baseline | Mean IoU |
| --- | --- |
| Per-edge mean box (predict the average box for each side) | 0.4295 |
| v2, direct regression, best held-out result | 0.4675 |

The per-edge mean is the number to keep in mind. Because the vertical axis is
nearly constant and the boxes are similarly sized, simply predicting the
average box for the correct side already scores 0.43 — so an IoU of 0.47 is a
much smaller achievement than it sounds, which is exactly what
[`experiments.md`](experiments.md) is about.

## Submission format

```csv
id,xmin,ymin,xmax,ymax,direction
image_1,-200.0,220.0,-120.0,380.0,0
```

Row order must match `submission_example.csv` exactly; `write_submission`
enforces that, along with rejecting NaN coordinates and missing ids. `direction`
accepts `-1` as "no prediction", which turned out to matter a great deal —
see [`method.md`](method.md#7-abstaining-on-direction).
