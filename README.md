# Detection by Shadow

**Find a pedestrian who is entirely outside the camera frame, using the shadow
they cast into it.**

[![CI](https://github.com/alex-krasnoshtanov/Detection-by-Shadow/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-krasnoshtanov/Detection-by-Shadow/actions/workflows/ci.yml)
[![Container](https://github.com/alex-krasnoshtanov/Detection-by-Shadow/actions/workflows/docker.yml/badge.svg)](https://github.com/alex-krasnoshtanov/Detection-by-Shadow/actions/workflows/docker.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Winning entry in the DEMCON Deep Tech track at BrabantHack 2026, built by a
three-person team.

A vehicle-mounted camera sees a shadow stretching in from the edge of the
frame. The person casting it is off-screen and may be about to step into the
road. The task is to predict their bounding box, in coordinates that fall
outside the image, plus whether they are walking into frame.

![Decomposing an off-frame bounding box](assets/decomposition.svg)

## Result

| | Mean IoU | Measured on |
| --- | --- | --- |
| Predict the average box per side | 0.4295 | 254-frame holdout (the floor) |
| Direct box regression, fully tuned, 120 epochs | 0.4675 | 203-frame holdout, 2026 |
| **Decomposed targets, 40 epochs + TTA** | **0.6096** | 254-frame holdout, reproduced here |
| Decomposed targets, 3-seed ensemble | 0.626 | *the organisers' hidden test set* |

Side classification is **254/254** on that held-out split. Direction peaks at
0.606 against a 48.3% base rate and abstains below 0.6 confidence, which on the
test set means declining on 407 of 414 frames.

The 0.626 was the team's winning leaderboard score on the organisers' hidden
test set. Development was collaborative, so treat it as the team's number.
Every other row is mean IoU on a 15% stratified holdout, measured by the code
in this repository.
[`docs/experiments.md`](docs/experiments.md) has the full run log.

![Predicted versus actual position of an off-frame pedestrian](assets/predictions.png)

Eight frames the model never trained on, taken from the validation split of the
run that produced the published weights. In every panel both boxes sit almost
entirely outside the dashed camera frame, which is the whole difficulty of the
task. The eight step evenly across the distance range, so the weak cases are in
there: accuracy is worst on the boxes closest to the frame, where the width is
hardest to pin down.

## The one idea that mattered

Every annotated person is *fully* outside the frame. `xmin` reaches −0.50 of the
image width and `xmax` reaches +1.44. Regressing those four corners directly
asks the network for coordinates on a scale that nothing in its input anchors.

We tuned that version properly first: ResNet-50, aspect-preserving input, GIoU
loss, a frozen-then-unfrozen trunk, photometric augmentation, flip TTA. It
reached 0.4675 mean IoU, against 0.4295 for simply predicting the average box
per side. All of that machinery bought 0.038.

So describe the box relative to the edge it hides behind instead:

| Output | | |
| --- | --- | --- |
| `side` | classification | which edge; trivial, and it saturates at 100% |
| `distance_from_edge` | regression, px | the actual difficulty |
| `bbox_width`, `bbox_height` | regression, px | |
| `y_center` | regression, px | nearly constant (σ = 14 px), so nearly free |

Same backbone, same data, same loss family. This version passes the tuned
direct-regression score in **three epochs** and reaches **0.6096** in forty.
Both of those figures were measured for this repository; the hackathon runs
never computed a local IoU for the decomposed model, so there was nothing to
quote. [The run log is
here](docs/experiments.md#filling-in-the-missing-comparison).

The change moved the project from marginally better than the mean to a
competitive score, and it brought three things along with it. The hard 1-D
problem separates cleanly from the easy 1-bit one. All four regression targets
are invariant under horizontal flip, so mirror TTA needs no coordinate fix-up
anywhere. And averaging an ensemble in this space cannot produce a box that
belongs to neither member.

Nineteen hand-crafted shadow descriptors go in alongside the image: where the
dark region sits, how its mass splits left and right, which way its principal
axis leans, how deep it is relative to the surrounding road. They are fused
onto the pooled trunk features through a 512-unit bottleneck, which restores
some of the spatial layout that global average pooling discards.

Full write-up: [`docs/method.md`](docs/method.md).

## Quickstart

[uv](https://docs.astral.sh/uv/) is the shortest path in:

```bash
git clone https://github.com/alex-krasnoshtanov/Detection-by-Shadow
cd Detection-by-Shadow
uv sync --extra dev
uv run pytest                # 211 tests, no dataset and no GPU needed
```

Use `uv sync --all-extras` instead if you want the demo and notebook
dependencies in the same environment. If you would rather use pip,
`pip install -e ".[dev]"` in a virtualenv gets you the same thing, and every
command below works without its `uv run` prefix.

### Try it in a browser

The lowest-friction way in. No dataset, no GPU, no training:

```bash
uv run --extra demo uvicorn shadow_detection.demo.app:app --port 8000
```

Then open `localhost:8000`. On a machine with no GPU, put
`UV_TORCH_BACKEND=cpu` in front of that command to pull the CPU torch wheel,
which is about 2 GB smaller than the CUDA one.

Or skip the clone entirely:

```bash
docker run --rm -p 8000:8000 -v shadow-models:/models \
  ghcr.io/alex-krasnoshtanov/detection-by-shadow:latest
```

Drop in a road frame and the predicted box appears on a canvas extended past
the image, since the box lies outside the picture. The model is pulled from the
release on first start and cached afterwards, which keeps 93 MB out of the git
history and makes every later start instant. One Python process serves both the
API and the page, so there is no Node toolchain and no build step. Details in
[`docs/demo.md`](docs/demo.md).

### Train it

The dataset is MIT-licensed, but at roughly 1 GB it is not vendored here. See
[`docs/dataset.md`](docs/dataset.md) for the layout the loader expects. With the
data in place, the best result reproduces in about 40 minutes on one modern GPU:

```bash
uv run shadow-detection train \
  --train-dir data/train_data/train_data \
  --preset ensemble \
  --output-dir runs/v5-ensemble \
  --batch-size 64 \
  --features-cache runs/features.npz

uv run shadow-detection predict \
  --test-dir data/test_data/test_data \
  --sample-csv data/submission_example.csv \
  --checkpoints runs/v5-ensemble/model_seed*.pt \
  --output runs/submission.csv
```

`--batch-size 64` because the preset's 128 was set on a 48 GB card. On 12 GB,
96 and above spill into host memory and cost 11x the wall-clock; 64 holds full
throughput at 8.8 GB. Measurements in
[`docs/experiments.md`](docs/experiments.md#batch-size-is-the-one-setting-you-may-have-to-change).

### Or run the released weights from the command line

Same model as the browser demo, batch inference straight to a submission CSV:

```bash
gh release download weights-v1 --repo alex-krasnoshtanov/Detection-by-Shadow
tar -xzf model_artifacts.tar.gz          # -> model.pt, target_stats.json

uv run shadow-detection predict \
  --test-dir path/to/frames \
  --sample-csv results/submission_example.csv \
  --checkpoints model.pt \
  --target-stats target_stats.json \
  --output submission.csv
```

`--checkpoints` accepts either form: a `state_dict` from `train`, or a
TorchScript archive like the released one. That is deliberate. A deployment
should not have to install the training package just to load a model, so the
released artifact carries its own graph, and
[`load_for_inference`](src/shadow_detection/model.py) works out which of the two
it was handed.

No dataset to hand at all? The descriptors run on any road photograph:

```bash
uv run shadow-detection features path/to/frame.png --mirrored
```

`train` writes a self-contained run directory: weights, `target_stats.json`
(the standardisation constants inference needs) and `run.json` (full config,
per-epoch history, wall-clock). `predict` ensembles every checkpoint it is
given and validates the submission's id order before writing it. `blend`
weighted-averages finished submissions. `export` traces a checkpoint to
TorchScript together with the stats file it is useless without.

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
  cli.py         shadow-detection {train,predict,blend,export,features}
  demo/          FastAPI app + static page, weights pulled from a release

Dockerfile       CPU-only image, published to GHCR by CD
docs/            method, experiment log, dataset description, demo
notebooks/       the five as-run hackathon notebooks, outputs preserved
explorations/    a classical + SAM3 pipeline, tried and dropped
results/         the submission CSVs that survive locally
tests/           211 tests, including a CPU train→predict→blend round trip
```

The notebooks are kept as archives. They hold the training logs that the
numbers in the docs are quoted from, complete with server paths and one
preserved traceback. The package is that work rewritten: the same method with
the paths as arguments, the mirror map in one place instead of three, and the
standardisation constants finally written to disk.

## What this does not show

The results flatter the method. Here is how:

- **The frames are synthetic renders** with consistent lighting, clean shadows
  and no occlusion or clutter. Side classification hitting 100% says more about
  how legible a raytraced shadow is than about real dashcam footage.
- **1692 training images.** Enough to overfit a 25M-parameter model within
  25 epochs, and nowhere near enough to learn walking direction.
- **Direction never became learnable.** Two architectures, two training
  regimes, both at chance against a 48.3% base rate. The submissions abstain
  instead: the format accepts `-1`, and a coin flip scores worse than
  declining.
- **The feature block was never ablated.** `ShadowNet(num_features=0)` builds
  the image-only comparison and is tested, but no run has measured it. The
  v2→v4 jump changed three things at once.
- **0.626 is the highest score with evidence behind it.** A blend across the
  team's models was attempted in the final hour; that notebook cell failed and
  no result was recorded.
- **0.6096 is one seed on one split.** No sweep, no cross-validation, and the
  254 held-out frames come from the same synthetic distribution as the training
  set.
- **Feature thresholds were chosen by eye** and never swept.

[`docs/method.md#known-issues`](docs/method.md#known-issues) has the rest,
including one real bug: the principal-axis angle is not negated when the
features are mirrored. It survives because the published scores were produced
with it, and a test pins the behaviour so that nobody fixes it silently.

## Credits

A three-person team: [Filipp Lotsmanov](https://github.com/filipp-lotsmanov),
Oleksii Krasnoshtanov and Danil Sysenko.

Collaborative in the way hackathons are. Notebooks passed between us, and
whoever's run scored best became everyone's starting point. The decomposed
target formulation, the 19 geometric descriptors and the flip-aware
augmentation all came out of that loop, and **0.626 is the team's result.**

What this repository adds on top of the shared work: the exploratory analysis
and the v1/v2 direct-regression line that diagnosed the target-space problem,
the full-resolution and all-data ensemble runs, and this rewrite into a tested
package.

Filipp has published his own take on the deployable half, a FastAPI + Next.js
demo:
**[filipp-lotsmanov/shadow-detection](https://github.com/filipp-lotsmanov/shadow-detection)**.
Worth a look alongside this one. His released weights are also how this
package's inference path was first cross-checked, back before it had trained
anything of its own.

Original hackathon repository (university account, full commit history):
[OleksiiKrasnoshtanov240247/Hackaton](https://github.com/OleksiiKrasnoshtanov240247/Hackaton).
Filipp's parallel exploratory notebook lives there and is not reproduced here.

## Licence

[MIT](LICENSE), matching the original repository. The challenge dataset is
MIT-licensed by its authors and is not included here.
