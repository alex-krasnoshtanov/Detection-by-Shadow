from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from istd_dataset import DatasetSample, compute_mask_metrics, load_mask, resolve_samples

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DIRECTION_LABELS = ["North", "NE", "East", "SE", "South", "SW", "West", "NW"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    mode: str
    input_dir: Path
    dataset_root: Path | None
    dataset_id: str
    dataset_split: str
    output_mask_dir: Path
    output_overlay_dir: Path
    output_prediction_dir: Path
    model_id: str
    text_prompt: str
    limit: int
    shuffle: bool
    seed: int
    use_sam: bool
    instance_threshold: float
    query_score_threshold: float
    query_top_k: int
    max_candidate_ratio: float
    min_prior_overlap: float
    max_side: int
    mask_threshold: float
    min_shadow_score: float
    min_shadow_area: int
    include_semantic_candidates: bool
    use_classical_fallback: bool
    debug_candidates: bool
    num_threads: int
    device: str


def prepare_assets(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    input_dir = base_dir / "assets" / "input"
    output_mask_dir = base_dir / "assets" / "output" / "masks"
    output_overlay_dir = base_dir / "assets" / "output" / "overlays"
    output_prediction_dir = base_dir / "assets" / "output" / "predictions"
    for d in (input_dir, output_mask_dir, output_overlay_dir, output_prediction_dir):
        d.mkdir(parents=True, exist_ok=True)
    return input_dir, output_mask_dir, output_overlay_dir, output_prediction_dir


def parse_args(base_dir: Path) -> InferenceConfig:
    default_input, default_masks, default_overlays, default_predictions = prepare_assets(base_dir)
    p = argparse.ArgumentParser(
        description="Shadow segmentation and off-screen person bbox estimation.",
    )

    # --- mode / data source ---
    p.add_argument("--mode", type=str, choices=["folder", "istd"], default="folder")
    p.add_argument("--input-dir", type=Path, default=default_input)
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="Local ISTD dataset root. Omit to auto-download via kagglehub.")
    p.add_argument("--dataset-id", type=str, default="sabarinathan/istd-dataset",
                   help="Kaggle dataset id for auto-download in --mode istd.")
    p.add_argument("--dataset-split", type=str, choices=["train", "test", "both"], default="both")

    # --- output dirs ---
    p.add_argument("--output-mask-dir", type=Path, default=default_masks)
    p.add_argument("--output-overlay-dir", type=Path, default=default_overlays)
    p.add_argument("--output-prediction-dir", type=Path, default=default_predictions)

    # --- sampling ---
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N images. 0 = all.")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # --- SAM (optional, off by default) ---
    p.add_argument("--use-sam", action="store_true",
                   help="Enable SAM3 as a refinement layer on top of classical detection.")
    p.add_argument("--model-id", type=str,
                   default=os.getenv("SAM_MODEL_ID", "facebook/sam3"))
    p.add_argument("--text-prompt", type=str, default="shadow")
    p.add_argument("--instance-threshold", type=float, default=0.45)
    p.add_argument("--query-score-threshold", type=float, default=0.25)
    p.add_argument("--query-top-k", type=int, default=16)
    p.add_argument("--max-candidate-ratio", type=float, default=0.75)
    p.add_argument("--min-prior-overlap", type=float, default=0.10)
    p.add_argument("--max-side", type=int, default=512,
                   help="Resize larger dimension to this before inference. 0 = no resize.")
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--min-shadow-score", type=float, default=10.0)
    p.add_argument("--min-shadow-area", type=int, default=300)
    p.add_argument("--include-semantic-candidates", action="store_true")
    p.add_argument("--no-classical-fallback", action="store_true")
    p.add_argument("--debug-candidates", action="store_true")

    # --- runtime ---
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"])

    args = p.parse_args()

    for d in (args.input_dir, args.output_mask_dir, args.output_overlay_dir,
              args.output_prediction_dir):
        d.mkdir(parents=True, exist_ok=True)

    return InferenceConfig(
        mode=args.mode,
        input_dir=args.input_dir,
        dataset_root=args.dataset_root,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        output_mask_dir=args.output_mask_dir,
        output_overlay_dir=args.output_overlay_dir,
        output_prediction_dir=args.output_prediction_dir,
        model_id=args.model_id,
        text_prompt=args.text_prompt.strip(),
        limit=max(0, args.limit),
        shuffle=args.shuffle,
        seed=args.seed,
        use_sam=args.use_sam,
        instance_threshold=args.instance_threshold,
        query_score_threshold=args.query_score_threshold,
        query_top_k=max(1, args.query_top_k),
        max_candidate_ratio=float(np.clip(args.max_candidate_ratio, 0.05, 0.98)),
        min_prior_overlap=float(np.clip(args.min_prior_overlap, 0.0, 1.0)),
        max_side=max(0, args.max_side),
        mask_threshold=args.mask_threshold,
        min_shadow_score=args.min_shadow_score,
        min_shadow_area=max(1, args.min_shadow_area),
        include_semantic_candidates=args.include_semantic_candidates,
        use_classical_fallback=not args.no_classical_fallback,
        debug_candidates=args.debug_candidates,
        num_threads=max(0, args.num_threads),
        device=args.device,
    )


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_for_inference(
    image_rgb: np.ndarray, max_side: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    orig_h, orig_w = image_rgb.shape[:2]
    if max_side <= 0 or max(orig_h, orig_w) <= max_side:
        return image_rgb, (orig_w, orig_h)
    scale = max_side / max(orig_h, orig_w)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), (orig_w, orig_h)


def select_samples(
    samples: list[DatasetSample], config: InferenceConfig,
) -> list[DatasetSample]:
    out = list(samples)
    if config.shuffle and len(out) > 1:
        random.Random(config.seed).shuffle(out)
    if config.limit > 0:
        out = out[: config.limit]
    return out


# ---------------------------------------------------------------------------
# Shadow segmentation -- classical (primary path)
# ---------------------------------------------------------------------------


def classical_shadow_mask(image_rgb: np.ndarray, min_area: int = 300) -> np.ndarray:
    """
    Shadow detection in LAB color space.

    Uses two cues that reflect shadow physics:
      1. Luminance drop  -- shadows make surfaces darker (lower L).
      2. Chromaticity preservation -- shadows don't change surface color,
         only intensity.  Dark objects DO change chromaticity relative to
         their surroundings.

    The luminance threshold comes from Otsu on the L channel, which finds
    the best bimodal split between shadow and lit pixels.  When the L
    distribution is unimodal (e.g. dappled shadows with low contrast),
    Otsu gives a threshold that captures most of the image; we detect this
    and fall back to a mean-std based threshold.

    Why NOT a local ratio (L / local_mean_L < 0.85):
      Fails for large shadows.  The center of a 200 px shadow has
      local_mean ~ own_value, so ratio ~ 1.0.  Only boundary pixels show
      a low ratio.  Otsu avoids this by using a global threshold.

    Why LAB instead of HSV:
      HSV V = max(R,G,B), which conflates surface albedo differences with
      shadow darkness.  LAB L is perceptually uniform luminance, which is
      what shadows actually affect.
    """
    h, w = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]       # luminance,   0-255 in OpenCV LAB
    a_ch = lab[:, :, 1]    # green-red,   ~0-255 centered at 128
    b_ch = lab[:, :, 2]    # blue-yellow, ~0-255 centered at 128

    # -- step 1: dark pixels via Otsu on L -----------------------------------
    L_u8 = np.clip(L, 0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(
        L_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    dark_mask = L < otsu_thr

    # Fallback: if Otsu captures > 55 % of pixels the distribution is close
    # to unimodal (e.g. uniformly lit scene with subtle shadows).  Switch to
    # mean - 0.6 * std which is more conservative.
    if dark_mask.mean() > 0.55:
        thr = max(40.0, float(L.mean() - 0.6 * L.std()))
        dark_mask = L < thr

    # If almost nothing passes (< 3 %), shadows may be very subtle.
    # Relax by 10 % above Otsu.
    if dark_mask.mean() < 0.03:
        dark_mask = L < (otsu_thr * 1.10)

    # -- step 2: chromaticity consistency ------------------------------------
    # Shadows preserve surface color.  Dark *objects* have inherently
    # different chromaticity.  Compare each pixel's (a, b) to a large-scale
    # reference that averages over both shadow and lit parts of the same
    # surface region.
    sigma = max(h, w) / 5.0
    ref_a = cv2.GaussianBlur(a_ch, (0, 0), sigmaX=sigma)
    ref_b = cv2.GaussianBlur(b_ch, (0, 0), sigmaX=sigma)
    chroma_dist = np.sqrt((a_ch - ref_a) ** 2 + (b_ch - ref_b) ** 2)
    chroma_ok = chroma_dist < 20.0

    # -- step 3: blue-shift cue (soft, outdoor-specific) ---------------------
    # Outdoor shadows receive blue sky fill light.  Shifts pixel's b channel
    # (blue-yellow) toward blue relative to its wide neighborhood.
    blue_shifted = (ref_b - b_ch) > 1.5

    # -- combine --------------------------------------------------------------
    shadow = dark_mask & (chroma_ok | blue_shifted)

    # -- morphological cleanup ------------------------------------------------
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = shadow.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # -- keep components above min_area ---------------------------------------
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 1

    return result.astype(bool)


# ---------------------------------------------------------------------------
# Shadow geometry: per-component feature extraction
# ---------------------------------------------------------------------------


def describe_components(mask: np.ndarray) -> list[dict[str, Any]]:
    """
    Label each connected component in the binary shadow mask and extract
    geometric features: centroid, principal axis (PCA), tip/root endpoints,
    elongation, entry edge, and a heuristic importance score.

    Tip  = end of the shadow deepest INTO the frame (furthest from edge).
    Root = end closest to a frame edge (approximately where the person's
           feet project onto the ground).
    """
    if not mask.any():
        return []

    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    height, width = mask.shape[:2]
    components: list[dict[str, Any]] = []

    for comp_id in range(1, n_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < 5:
            continue

        component = labels == comp_id
        points = np.column_stack(np.where(component))  # (row, col)
        if points.shape[0] < 5:
            continue

        centroid_row, centroid_col = points.mean(axis=0)

        # PCA: principal axis direction + elongation
        centered = points - points.mean(axis=0)
        if points.shape[0] > 2:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            axis = eigenvectors[:, -1]
        else:
            eigenvalues = np.array([1.0, 1.0])
            axis = np.array([0.0, 1.0])

        # Project onto principal axis to find extreme points
        projections = centered @ axis
        tip_idx = int(np.argmax(projections))
        root_idx = int(np.argmin(projections))
        tip_row, tip_col = points[tip_idx]
        root_row, root_col = points[root_idx]

        # Disambiguate tip vs root: root should be closer to a frame edge
        # (that's where the person's feet are, just off-screen).
        # PCA gives an undirected axis, so we might have them swapped.
        root_edge_dist = min(root_row, height - 1 - root_row,
                             root_col, width - 1 - root_col)
        tip_edge_dist = min(tip_row, height - 1 - tip_row,
                            tip_col, width - 1 - tip_col)
        if tip_edge_dist < root_edge_dist:
            tip_row, tip_col, root_row, root_col = (
                root_row, root_col, tip_row, tip_col
            )

        # Direction angle: from tip toward root (toward person).
        # atan2(dx, -dy) gives compass-style angle where North = 0.
        dx = float(root_col - tip_col)
        dy = float(root_row - tip_row)
        angle_rad = float(np.arctan2(dx, -dy))

        elongation = float(
            np.log1p(float(eigenvalues[-1]) / (float(eigenvalues[0]) + 1e-6))
        )

        entry_distances = {
            "top":    int(root_row),
            "bottom": int(height - 1 - root_row),
            "left":   int(root_col),
            "right":  int(width - 1 - root_col),
        }
        entry_edge = min(entry_distances, key=entry_distances.get)

        # Heuristic score: bigger + more elongated = more likely a person shadow
        score = float(area * (1.0 + 0.35 * elongation))

        components.append({
            "component_id": comp_id,
            "area": area,
            "centroid_x": float(centroid_col),
            "centroid_y": float(centroid_row),
            "root_x": float(root_col),
            "root_y": float(root_row),
            "tip_x": float(tip_col),
            "tip_y": float(tip_row),
            "angle_rad": angle_rad,
            "angle_cos": float(np.cos(angle_rad)),
            "angle_sin": float(np.sin(angle_rad)),
            "elongation": elongation,
            "entry_edge": entry_edge,
            "score": score,
            "bbox": [
                int(stats[comp_id, cv2.CC_STAT_LEFT]),
                int(stats[comp_id, cv2.CC_STAT_TOP]),
                int(stats[comp_id, cv2.CC_STAT_LEFT] + stats[comp_id, cv2.CC_STAT_WIDTH]),
                int(stats[comp_id, cv2.CC_STAT_TOP] + stats[comp_id, cv2.CC_STAT_HEIGHT]),
            ],
        })

    components.sort(key=lambda c: c["score"], reverse=True)
    return components


# ---------------------------------------------------------------------------
# Direction classification
# ---------------------------------------------------------------------------


def angle_to_direction_class(angle_rad: float) -> int:
    n = len(DIRECTION_LABELS)
    sector = 2.0 * np.pi / n
    angle_norm = angle_rad % (2.0 * np.pi)
    return int((angle_norm + sector / 2.0) // sector) % n


# ---------------------------------------------------------------------------
# Off-screen bbox prediction (geometry-based heuristic)
# ---------------------------------------------------------------------------


def predict_shadow_target(
    image_rgb: np.ndarray, shadow_mask: np.ndarray,
) -> dict[str, Any]:
    """
    Given a shadow mask, estimate the off-screen bounding box of the person
    casting the primary shadow and their walking direction.

    Uses the shadow's root point (near frame edge), tip-to-root vector
    (direction toward person), shadow area (proxy for person distance / size),
    and entry edge to project a bbox outside the frame.
    """
    components = describe_components(shadow_mask)
    height, width = image_rgb.shape[:2]

    if not components:
        return {
            "bbox": [
                float(-0.15 * width), float(-0.2 * height),
                float(0.15 * width),  float(0.2 * height),
            ],
            "direction_class": 0,
            "direction_label": DIRECTION_LABELS[0],
            "components": [],
            "primary_component": None,
        }

    primary = components[0]

    root = np.array([primary["root_x"], primary["root_y"]], dtype=np.float64)
    tip = np.array([primary["tip_x"], primary["tip_y"]], dtype=np.float64)
    toward_person = root - tip
    distance = float(np.linalg.norm(toward_person))

    if distance < 1e-6:
        toward_person = np.array([0.0, -1.0])
        distance = 1.0

    unit = toward_person / distance

    # Offset: how far from the root to place the bbox center.
    # Scaled by shadow length and area (longer shadow = further person).
    offset = max(24.0, 0.45 * distance, 0.15 * np.sqrt(primary["area"]))
    center = root + unit * offset

    # Push the center past the frame edge if entry_edge says the shadow
    # enters from that side.
    edge_margin = max(16.0, 0.35 * offset)
    edge = primary["entry_edge"]
    if edge == "left":
        center[0] = min(center[0], -edge_margin)
    elif edge == "right":
        center[0] = max(center[0], width + edge_margin)
    elif edge == "top":
        center[1] = min(center[1], -edge_margin)
    elif edge == "bottom":
        center[1] = max(center[1], height + edge_margin)

    # Person bbox size: rough proportions from shadow area and offset.
    person_h = max(44.0, offset * 0.9, np.sqrt(primary["area"]) * 0.95)
    person_w = max(24.0, person_h * 0.38)
    bbox = [
        float(center[0] - person_w / 2.0),
        float(center[1] - person_h / 2.0),
        float(center[0] + person_w / 2.0),
        float(center[1] + person_h / 2.0),
    ]

    direction_class = angle_to_direction_class(primary["angle_rad"])

    return {
        "bbox": bbox,
        "direction_class": direction_class,
        "direction_label": DIRECTION_LABELS[direction_class],
        "components": components,
        "primary_component": primary,
    }


# ---------------------------------------------------------------------------
# Prediction rendering
# ---------------------------------------------------------------------------


def render_prediction_canvas(
    image_rgb: np.ndarray,
    shadow_mask: np.ndarray,
    prediction: dict[str, Any],
) -> np.ndarray:
    """
    Draw the predicted off-screen bbox, shadow mask overlay, and direction
    arrow on an extended canvas so the off-screen region is visible.
    """
    height, width = image_rgb.shape[:2]
    pad_x = max(48, int(width * 0.35))
    pad_y = max(48, int(height * 0.35))

    canvas = np.full((height + 2 * pad_y, width + 2 * pad_x, 3), 40, dtype=np.uint8)
    canvas[pad_y:pad_y + height, pad_x:pad_x + width] = image_rgb

    # Frame boundary
    cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + width, pad_y + height),
                  (220, 220, 220), 1)

    # Shadow mask overlay
    mask_bool = shadow_mask.astype(bool)
    if mask_bool.any():
        region = canvas[pad_y:pad_y + height, pad_x:pad_x + width].copy()
        highlight = np.array([255, 110, 0], dtype=np.float32)
        region[mask_bool] = (
            region[mask_bool].astype(np.float32) * 0.5 + highlight * 0.5
        ).astype(np.uint8)
        canvas[pad_y:pad_y + height, pad_x:pad_x + width] = region

    primary = prediction.get("primary_component")
    if primary:
        root_pt = (int(round(primary["root_x"])) + pad_x,
                   int(round(primary["root_y"])) + pad_y)
        tip_pt = (int(round(primary["tip_x"])) + pad_x,
                  int(round(primary["tip_y"])) + pad_y)
        cv2.circle(canvas, root_pt, 5, (0, 220, 255), -1)   # yellow-ish: root
        cv2.circle(canvas, tip_pt, 5, (255, 180, 0), -1)     # blue-ish: tip
        cv2.line(canvas, tip_pt, root_pt, (0, 220, 255), 2)

    # Predicted bbox
    bbox = prediction["bbox"]
    x1 = int(round(bbox[0])) + pad_x
    y1 = int(round(bbox[1])) + pad_y
    x2 = int(round(bbox[2])) + pad_x
    y2 = int(round(bbox[3])) + pad_y
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 80, 255), 2)

    # Direction arrow: root -> bbox center
    if primary:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        root_pt = (int(round(primary["root_x"])) + pad_x,
                   int(round(primary["root_y"])) + pad_y)
        cv2.arrowedLine(canvas, root_pt, (cx, cy), (0, 200, 100), 2, tipLength=0.15)

    label = prediction["direction_label"]
    cv2.putText(canvas, f"Dir: {label}", (pad_x + 6, pad_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Orange=shadow  Red=predicted bbox",
                (pad_x + 6, pad_y + height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# SAM3 (optional refinement, off by default)
# ---------------------------------------------------------------------------


def load_sam_model(model_id: str, device: str):
    """Load SAM3 model and processor from HuggingFace."""
    from transformers import Sam3Model, Sam3Processor

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        processor = Sam3Processor.from_pretrained(model_id, token=token)
        model = Sam3Model.from_pretrained(model_id, token=token, dtype=dtype)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SAM model '{model_id}'. "
            "Check model id, HF token, and that torchvision is installed. "
            f"Original error: {exc}"
        ) from exc

    return model, processor


def build_text_cache(model, processor, text_prompt: str, device: str) -> dict[str, Any]:
    prompt = text_prompt.strip() or "shadow"
    text_inputs = processor(text=prompt, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    return {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs.get("attention_mask"),
    }


def infer_candidate_masks(
    model, processor, config: InferenceConfig,
    image_rgb: np.ndarray, text_cache: dict[str, Any], device: str,
) -> list[np.ndarray]:
    """Run SAM3 forward pass and collect candidate binary masks."""

    def _resize(m: np.ndarray, th: int, tw: int) -> np.ndarray:
        if m.shape == (th, tw):
            return m
        return cv2.resize(m.astype(np.float32), (tw, th),
                          interpolation=cv2.INTER_LINEAR)

    def _valid(m: np.ndarray) -> bool:
        r = float(m.mean())
        return 0.0008 <= r <= config.max_candidate_ratio

    pil_image = Image.fromarray(image_rgb)
    th, tw = image_rgb.shape[:2]

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=text_cache["input_ids"],
            attention_mask=text_cache["attention_mask"],
        )

    candidates: list[np.ndarray] = []

    # Query masks from the DETR decoder
    query_masks = torch.sigmoid(outputs.pred_masks[0]).cpu().numpy()
    query_scores = torch.sigmoid(outputs.pred_logits[0]).cpu().numpy()
    if outputs.presence_logits is not None:
        presence = float(torch.sigmoid(outputs.presence_logits[0]).cpu().item())
        query_scores = query_scores * presence

    top_ids = np.argsort(-query_scores)[: config.query_top_k]
    thr = float(np.clip(config.mask_threshold, 0.02, 0.98))
    for idx in top_ids:
        if float(query_scores[idx]) < config.query_score_threshold:
            continue
        resized = _resize(query_masks[idx], th, tw)
        binary = resized >= thr
        if _valid(binary):
            candidates.append(binary)

    # Optional semantic map candidates
    if config.include_semantic_candidates:
        semantic = torch.sigmoid(outputs.semantic_seg[0, 0]).cpu().numpy()
        semantic = _resize(semantic, th, tw)
        for st in [max(0.55, config.instance_threshold),
                    max(0.65, config.instance_threshold + 0.12),
                    max(0.75, config.instance_threshold + 0.22)]:
            binary = semantic >= st
            if _valid(binary):
                candidates.append(binary)

    # De-duplicate near-identical masks (IoU > 0.96)
    unique: list[np.ndarray] = []
    for cand in candidates:
        keep = True
        for prev in unique:
            inter = float(np.logical_and(cand, prev).sum())
            union = float(np.logical_or(cand, prev).sum()) + 1e-6
            if inter / union > 0.96:
                keep = False
                break
        if keep:
            unique.append(cand)

    return unique


def estimate_shadow_prior(image_rgb: np.ndarray) -> np.ndarray:
    """
    Rough prior mask: pixels that are locally darker, low-saturation, and
    below an adaptive intensity threshold.  Used to validate SAM candidates.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0

    sigma = max(9.0, max(image_rgb.shape[:2]) / 16.0)
    illum = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    shadow_idx = np.clip((illum - gray) / (illum + 1e-3), 0.0, 1.0)

    thr_shadow = max(0.08, float(np.percentile(shadow_idx, 65)))
    thr_dark = float(np.percentile(gray, 55))

    prior = (shadow_idx >= thr_shadow) & (gray <= thr_dark) & (sat <= 0.82)

    prior_u8 = prior.astype(np.uint8)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    prior_u8 = cv2.morphologyEx(prior_u8, cv2.MORPH_OPEN, k_open, iterations=1)
    prior_u8 = cv2.morphologyEx(prior_u8, cv2.MORPH_CLOSE, k_close, iterations=1)
    return prior_u8.astype(bool)


def score_shadow_candidate(mask: np.ndarray, image_rgb: np.ndarray) -> float:
    """
    Score a candidate mask by how shadow-like it is: high contrast with
    its immediate boundary, low internal saturation, reasonable area.
    """
    if not mask.any():
        return -1e9

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    area = int(mask.sum())
    inside_v = float(gray[mask].mean())
    inside_s = float(hsv[..., 1][mask].mean())

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & (~mask)
    if ring.sum() < 50:
        return -1e9
    outside_v = float(gray[ring].mean())

    contrast = outside_v - inside_v
    sat_penalty = max(0.0, inside_s - 85.0) * 0.25
    ratio = area / float(mask.shape[0] * mask.shape[1])
    area_penalty = max(0.0, ratio - 0.35) * 55.0
    area_boost = min(8.0, np.log1p(area) * 0.55)
    return contrast - sat_penalty + area_boost - area_penalty


def build_sam_shadow_mask(
    config: InferenceConfig,
    candidates: list[np.ndarray],
    image_rgb: np.ndarray,
) -> np.ndarray:
    """
    From SAM candidate masks, select and merge those that look shadow-like.
    Falls back to classical mask or prior when no good candidates remain.
    """
    prior = estimate_shadow_prior(image_rgb)
    classical = classical_shadow_mask(image_rgb, config.min_shadow_area)

    kept: list[tuple[float, np.ndarray]] = []
    for mask in candidates:
        if int(mask.sum()) < config.min_shadow_area:
            continue
        overlap = float(np.logical_and(mask, prior).sum()) / (float(mask.sum()) + 1e-6)
        if overlap < config.min_prior_overlap:
            continue
        score = score_shadow_candidate(mask, image_rgb)
        if config.debug_candidates:
            print(f"  area={int(mask.sum()):6d} overlap={overlap:.3f} "
                  f"score={score:.2f} kept={score >= config.min_shadow_score}")
        if score >= config.min_shadow_score:
            kept.append((score, mask))

    if not kept:
        if config.use_classical_fallback and classical.any():
            return classical.astype(np.uint8)
        return prior.astype(np.uint8)

    kept.sort(key=lambda t: t[0], reverse=True)
    selected = [m for _, m in kept[:25]]
    merged = np.any(np.stack(selected, axis=0), axis=0)

    # Intersect with dilated prior to remove noise
    dilated_prior = cv2.dilate(
        prior.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1,
    ).astype(bool)
    merged = merged & dilated_prior
    merged_u8 = merged.astype(np.uint8)

    # Guard: near-full-frame output is garbage
    if float(merged_u8.mean()) > 0.8:
        if config.use_classical_fallback and classical.any():
            return classical.astype(np.uint8)
        return prior.astype(np.uint8)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(merged_u8, cv2.MORPH_OPEN, k_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= config.min_shadow_area:
            filtered[labels == i] = 1

    # Final guard
    ratio = float(filtered.mean())
    if ratio > 0.8 or ratio < 0.001:
        if config.use_classical_fallback and classical.any():
            return classical.astype(np.uint8)
        return prior.astype(np.uint8)

    return (filtered > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def save_outputs(
    source_path: Path,
    image_rgb: np.ndarray,
    shadow_mask: np.ndarray,
    prediction: dict[str, Any],
    output_mask_dir: Path,
    output_overlay_dir: Path,
    output_prediction_dir: Path,
) -> None:
    stem = source_path.stem

    # Binary mask
    mask_u8 = (shadow_mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8).save(output_mask_dir / f"{stem}_shadow_mask.png")

    # Overlay: orange tint on detected shadow
    overlay = image_rgb.copy()
    mask_bool = shadow_mask.astype(bool)
    if mask_bool.any():
        highlight = np.array([255, 85, 0], dtype=np.float32)
        overlay[mask_bool] = (
            overlay[mask_bool].astype(np.float32) * 0.55 + highlight * 0.45
        ).astype(np.uint8)
    Image.fromarray(overlay).save(output_overlay_dir / f"{stem}_shadow_overlay.png")

    # Prediction canvas (extended, shows off-screen bbox)
    canvas = render_prediction_canvas(image_rgb, shadow_mask, prediction)
    Image.fromarray(canvas).save(output_prediction_dir / f"{stem}_shadow_prediction.png")


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------


def process_image(
    sample: DatasetSample,
    config: InferenceConfig,
    sam_model=None,
    sam_processor=None,
    text_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_rgb = load_image_rgb(sample.image_path)
    infer_rgb, original_size = resize_for_inference(original_rgb, config.max_side)

    # Primary path: classical shadow detection
    shadow_mask = classical_shadow_mask(infer_rgb, config.min_shadow_area)

    # Optional SAM refinement
    if (config.use_sam and sam_model is not None
            and sam_processor is not None and text_cache is not None):
        candidates = infer_candidate_masks(
            sam_model, sam_processor, config,
            infer_rgb, text_cache, config.device,
        )
        sam_mask = build_sam_shadow_mask(config, candidates, infer_rgb).astype(bool)

        classical_ratio = float(shadow_mask.mean())
        sam_ratio = float(sam_mask.mean())

        # Trust SAM when classical found nothing but SAM found something plausible
        if classical_ratio < 0.001 and 0.001 < sam_ratio < 0.65:
            shadow_mask = sam_mask
        # Otherwise merge: classical base + SAM additions near classical regions
        elif 0.001 < sam_ratio < 0.65:
            dilated = cv2.dilate(
                shadow_mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1,
            ).astype(bool)
            shadow_mask = shadow_mask | (sam_mask & dilated)

    # Resize mask back to original dimensions if needed
    if shadow_mask.shape[:2] != (original_size[1], original_size[0]):
        shadow_mask = cv2.resize(
            shadow_mask.astype(np.uint8),
            dsize=original_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    prediction = predict_shadow_target(original_rgb, shadow_mask)

    # Compare to ground truth if available
    gt_metrics: dict[str, float] | None = None
    if sample.mask_path is not None and sample.mask_path.exists():
        gt_mask = load_mask(sample.mask_path)
        gt_metrics = compute_mask_metrics(shadow_mask, gt_mask)

    save_outputs(
        source_path=sample.image_path,
        image_rgb=original_rgb,
        shadow_mask=shadow_mask,
        prediction=prediction,
        output_mask_dir=config.output_mask_dir,
        output_overlay_dir=config.output_overlay_dir,
        output_prediction_dir=config.output_prediction_dir,
    )

    return {
        "image": sample.image_path.name,
        "image_path": str(sample.image_path),
        "split": sample.split,
        "stem": sample.stem,
        "mask_coverage": float(shadow_mask.mean()),
        "prediction": prediction,
        "ground_truth_mask_path": (
            str(sample.mask_path) if sample.mask_path is not None else None
        ),
        "clean_image_path": (
            str(sample.clean_path) if sample.clean_path is not None else None
        ),
        "ground_truth_metrics": gt_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    config = parse_args(base_dir)

    # Resolve input samples (folder or ISTD dataset)
    if config.mode == "folder" and not list_images(config.input_dir):
        src = config.dataset_root or config.dataset_id
        print(f"No images in {config.input_dir}. Falling back to ISTD: {src}",
              file=sys.stderr)

    try:
        samples = resolve_samples(
            mode=config.mode,
            input_dir=config.input_dir,
            dataset_root=config.dataset_root,
            dataset_id=config.dataset_id,
            split=config.dataset_split,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    samples = select_samples(samples, config)
    if not samples:
        print(f"No images found for mode={config.mode}.", file=sys.stderr)
        sys.exit(1)

    # Thread configuration
    if config.device == "cpu" and config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
        torch.set_num_interop_threads(min(config.num_threads, 4))

    # Optional SAM model loading
    sam_model = None
    sam_processor = None
    text_cache = None

    if config.use_sam:
        try:
            sam_model, sam_processor = load_sam_model(config.model_id, config.device)
            text_cache = build_text_cache(
                sam_model, sam_processor, config.text_prompt, config.device,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)

    # Status
    print(f"Processing {len(samples)} image(s) on {config.device}...")
    if config.mode == "istd":
        src = config.dataset_root or config.dataset_id
        print(f"  ISTD mode: {src} ({config.dataset_split})")
    if config.use_sam:
        print(f"  SAM refinement: {config.model_id}")

    # Process
    reports: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Shadow segmentation"):
        try:
            report = process_image(
                sample, config, sam_model, sam_processor, text_cache,
            )
            reports.append(report)
        except RuntimeError as exc:
            print(f"[ERROR] {sample.image_path.name}: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[ERROR] {sample.image_path.name}: {exc}", file=sys.stderr)

    # Save report
    report_path = config.output_prediction_dir / "predictions.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2, default=str)

    # Print summary if GT metrics available
    metrics = [r["ground_truth_metrics"] for r in reports
               if r.get("ground_truth_metrics") is not None]
    if metrics:
        mean_iou = np.mean([m["iou"] for m in metrics])
        mean_dice = np.mean([m["dice"] for m in metrics])
        print(f"\nGround truth comparison ({len(metrics)} images):")
        print(f"  Mean IoU:  {mean_iou:.4f}")
        print(f"  Mean Dice: {mean_dice:.4f}")

    print("\nOutputs saved to:")
    print(f"  Masks:       {config.output_mask_dir}")
    print(f"  Overlays:    {config.output_overlay_dir}")
    print(f"  Predictions: {config.output_prediction_dir}")


if __name__ == "__main__":
    main()