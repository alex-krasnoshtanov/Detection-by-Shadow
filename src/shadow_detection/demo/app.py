"""A single-process web demo: FastAPI serves both the API and the page.

Deliberately one service rather than a separate API and frontend. The whole
value of the demo is that someone can clone the repository, run one command and
see a prediction; a second toolchain to install and a second port to configure
works against that. The page is plain HTML, CSS and JavaScript served as static
files, so there is no build step and the container is one Python image.

    uvicorn shadow_detection.demo.app:app --port 8000

The model is pulled from a GitHub release on first start and cached; see
:mod:`shadow_detection.demo.weights`.
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from shadow_detection.data import TargetStats
from shadow_detection.demo.weights import WeightsUnavailableError, ensure_weights
from shadow_detection.geometry import LEFT, FrameSize
from shadow_detection.model import load_for_inference
from shadow_detection.predict import (
    DEFAULT_DIRECTION_THRESHOLD,
    DIRECTION_ABSTAIN,
    predict_images,
    to_submission,
)

STATIC_DIR = Path(__file__).parent / "static"

#: Refuse oversized uploads before decoding them.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

#: Guard against a decompression bomb in an uploaded image.
MAX_PIXELS = 40_000_000

FRAME = FrameSize(720, 480)

#: Populated by the lifespan handler.
state: dict[str, Any] = {"model": None, "stats": None, "device": None, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once, at startup, rather than per request.

    A failure here is recorded rather than raised: the server still starts and
    ``/api/health`` explains what went wrong, which is far easier to debug in a
    container than a process that exits immediately.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_path, stats_path = ensure_weights()
        state["model"] = load_for_inference(model_path, device=device)
        state["stats"] = TargetStats.load(stats_path)
        state["device"] = str(device)
        print(f"model loaded on {device}")
    except (WeightsUnavailableError, OSError, ValueError) as error:
        state["error"] = str(error)
        print(f"model unavailable: {error}")
    yield
    state.clear()


app = FastAPI(
    title="Detection by Shadow",
    description="Locate an off-frame pedestrian from the shadow they cast into the frame.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> JSONResponse:
    """Whether the model is loaded, and why not if it isn't."""
    ready = state.get("model") is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "device": state.get("device"),
            "error": state.get("error"),
            "frame": {"width": FRAME.width, "height": FRAME.height},
            "direction_threshold": DEFAULT_DIRECTION_THRESHOLD,
        },
    )


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)) -> dict:  # noqa: B008 - FastAPI idiom
    """Predict the off-frame box for one uploaded frame.

    Coordinates come back in the *uploaded image's* pixel space. The model
    reasons in the 720x480 frame it was trained on, so a differently sized
    upload is scaled on the way out -- otherwise the box would be drawn in the
    wrong place by exactly the aspect mismatch.
    """
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail=state.get("error") or "model not loaded")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image is {len(payload) / 1e6:.1f} MB, limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB",
        )

    try:
        frame = Image.open(io.BytesIO(payload))
        frame.load()
        frame = frame.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail=f"could not decode image: {error}") from error

    width, height = frame.size
    if width * height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"image is {width}x{height}, too large")

    started = time.perf_counter()
    raw = predict_images(
        state["model"], [frame], names=[image.filename or "upload"], device=state["device"]
    )[0]
    row = to_submission([raw], state["stats"], FRAME).iloc[0]
    elapsed_ms = (time.perf_counter() - started) * 1000

    # Model coordinates are in the training frame; rescale to the upload.
    scale_x = width / FRAME.width
    scale_y = height / FRAME.height

    direction = int(row["direction"])
    return {
        "bbox": {
            "xmin": float(row["xmin"]) * scale_x,
            "ymin": float(row["ymin"]) * scale_y,
            "xmax": float(row["xmax"]) * scale_x,
            "ymax": float(row["ymax"]) * scale_y,
        },
        "side": int(raw.side_probs.argmax()),
        "side_label": "left" if int(raw.side_probs.argmax()) == LEFT else "right",
        "side_confidence": float(raw.side_probs.max()),
        "direction": direction,
        "direction_label": {
            DIRECTION_ABSTAIN: "abstained",
            0: "out of frame",
            1: "into frame",
        }[direction],
        "direction_confidence": float(raw.direction_probs.max()),
        "image_width": width,
        "image_height": height,
        "inference_ms": round(elapsed_ms, 1),
        "device": state["device"],
        "tta": True,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
