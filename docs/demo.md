# The web demo

Drop a road frame in, get the predicted off-frame box drawn on an extended
canvas. One process: FastAPI serves the JSON API and the page.

```bash
uv run --extra demo uvicorn shadow_detection.demo.app:app --port 8000
# open http://localhost:8000
```

With pip, that is `pip install -e ".[demo]"` followed by the same `uvicorn`
line. On a machine with no GPU, put `UV_TORCH_BACKEND=cpu` in front of the sync
to get the CPU torch wheel and save about 2 GB.

The model is not in git. It is 93 MB of binary that would weigh on every clone,
so it is fetched from a GitHub release on first start and cached; later starts
are instant and work offline.

## Why one service and not two

An API plus a separate single-page app is the conventional split, and it buys
nothing here. The entire value of a demo is that someone can clone the
repository, run one command and see a prediction. A second toolchain to
install, a second port to configure and a CORS policy to get wrong all work
against that. The page is plain HTML, CSS and JavaScript served as static files: no
build step, no `node_modules`, and the container is one Python image.

## The drawing problem

The predicted box is mostly *outside* the uploaded image, so the canvas cannot
just be the image. It is sized to the union of the frame and the box, with the
image drawn at an offset inside it and the frame edge dashed so it is obvious
the person is beyond it. That is the same reason
[`assets/predictions.png`](../assets/predictions.png) is drawn on an extended
canvas — you cannot show this prediction inside the picture it came from.

Coordinates come back in the *uploaded image's* pixel space. The model reasons
in the 720x480 frame it was trained on, so anything else is rescaled on the way
out; skip that and the box lands wrong by exactly the scale factor.

## Configuration

| Variable | Effect |
| --- | --- |
| `SHADOW_MODEL_DIR` | where to cache the model (default: platform cache dir) |
| `SHADOW_MODEL_URL` | full URL of the archive to fetch |
| `SHADOW_RELEASE_TAG` | release tag to build the default URL from |
| `SHADOW_MODEL_SHA256` | expected digest, overriding the built-in one |

Pointing the demo at a model you just trained, without publishing anything:

```bash
uv run shadow-detection export runs/v5-ensemble/model_seed42.pt -o local/model.pt
SHADOW_MODEL_DIR=local uv run --extra demo uvicorn shadow_detection.demo.app:app --port 8000
```

`export` copies `target_stats.json` next to the traced model, because a trace
on its own cannot turn its regression outputs back into pixels.

## Publishing weights

[`scripts/package_release.py`](../scripts/package_release.py) builds the archive
the loader above expects, and prints the digest to pin and the `gh` command to
publish it:

```bash
uv run python scripts/package_release.py runs/v5-ensemble/model_seed42.pt
```

It refuses to run without the run's `target_stats.json`, since an archive
carrying only the trace produces a demo that loads happily and predicts
nonsense. Pinning `EXPECTED_SHA256` in
[`weights.py`](../src/shadow_detection/demo/weights.py) after publishing turns a
corrupted or substituted download into an error instead of a bad prediction.

## API

`POST /api/predict` takes a multipart image and returns:

```json
{
  "bbox": { "xmin": -234.4, "ymin": 223.4, "xmax": -120.5, "ymax": 413.4 },
  "side": 0,
  "side_label": "left",
  "side_confidence": 0.999,
  "direction": -1,
  "direction_label": "abstained",
  "direction_confidence": 0.529,
  "image_width": 720,
  "image_height": 480,
  "inference_ms": 61.4,
  "device": "cuda",
  "tta": true
}
```

`side`: 0 = off-frame left, 1 = off-frame right. `direction`: 1 = into frame,
0 = out of frame, −1 = abstained. Direction abstains below 0.6 confidence and
usually does; see [method.md](method.md#7-abstaining-on-direction).

`GET /api/health` reports whether the model loaded and, if not, why. Full
OpenAPI at `/docs`.

Latency with flip TTA, measured on an RTX 5070: about 60–80 ms warm, and
several hundred on the very first request while CUDA initialises.

## Container

```bash
docker run --rm -p 8000:8000 -v shadow-models:/models \
  ghcr.io/alex-krasnoshtanov/detection-by-shadow:latest
```

CPU-only wheels, so it runs anywhere. Mount a volume at `/models` and the
weights survive restarts rather than being re-downloaded. Published to GHCR by
[`.github/workflows/docker.yml`](../.github/workflows/docker.yml), which builds
the image, boots it, and checks the API and static assets respond *before*
pushing — the failure it exists to catch is a slimmed dependency list that
breaks a module-scope import, which nothing else would notice until someone
pulled the image.

## Limits

The upload cap is 12 MB and 40 megapixels, both rejected before decoding.
Uploads are held in memory, never written to disk. There is no rate limiting and
no authentication. Run it locally, or behind something that provides both.
