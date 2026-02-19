"""
SAM3 Distributed Segmentation Inference - Native Batching Version (IMAGE-SHARDED)
WITH MASK-LEVEL NMS STITCHING

Changes from inference_v5 (pixel-level max confidence):
  - REMOVED: stitch_masks_for_prompt_max_confidence()
  - REMOVED: save_combined_mask_max_confidence()
  - REMOVED: save_overlay()
  - ADDED:   mask_to_rle / rle_to_mask / rle_intersection_area  (RLE helpers)
  - ADDED:   collect_global_masks()   -- project patch masks → full image RLE
  - ADDED:   greedy_nms_masks()       -- object-level deduplication
  - ADDED:   rasterize_per_class / rasterize_combined / rasterize_overlay
  - CHANGED: process_and_save_one_image_single_rank() -- new post-inference block
  - CHANGED: main() -- two new CLI args

Author: ALS Photon Science Computing
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

if "HF_HUB_CACHE" in os.environ:
    os.environ["HF_HOME"] = os.environ["HF_HUB_CACHE"]

import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GLOBAL_COUNTER = 1

CLASS_COLORS_RGB = {
    "Cortex": (0.0, 0.0, 1.0),
    "Phloem Fibers": (0.0, 1.0, 0.0),
    "Phloem": (128/255, 0.0, 128/255),
    "Xylem vessels": (1.0, 0.0, 0.0),
    "Air-based Pith cells": (1.0, 1.0, 0.0),
    "Water-based Pith cells": (1.0, 165/255, 0.0),
}


# =============================================================================
# SAM3 Native API Helper Functions  (UNCHANGED)
# =============================================================================
def create_empty_datapoint() -> Datapoint:
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint: Datapoint, pil_image: Image.Image) -> None:
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint: Datapoint, text_query: str) -> int:
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "Please set the image first"
    h, w = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


# =============================================================================
# Image Processing Utilities  (UNCHANGED)
# =============================================================================
def load_state_dict_flexible(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def tif_to_rgb_pil(
    image_path: str,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> Image.Image:
    img = Image.open(image_path)
    if img.mode not in ["RGB", "L", "RGBA"]:
        if img.mode in ["I", "I;16", "F"]:
            img_array = np.array(img).astype(np.float32)
            p_low = np.percentile(img_array, percentile_low)
            p_high = np.percentile(img_array, percentile_high)
            img_array = np.clip(img_array, p_low, p_high)
            denom = (p_high - p_low) if (p_high - p_low) > 1e-6 else 1.0
            img_array = ((img_array - p_low) / denom * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        else:
            img = img.convert("L")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def crop_image_to_patches(
    image_pil: Image.Image,
    patch_size: int = 512,
    overlap_ratio: float = 0.25,
) -> Tuple[List[Tuple], int, int, int, int]:
    W, H = image_pil.size
    patches: List[Tuple] = []
    step = int(patch_size * (1 - overlap_ratio))
    n_rows = max(1, (H - patch_size) // step + 1)
    n_cols = max(1, (W - patch_size) // step + 1)
    if n_rows > 1 and (n_rows - 1) * step + patch_size < H:
        n_rows += 1
    if n_cols > 1 and (n_cols - 1) * step + patch_size < W:
        n_cols += 1
    for row in range(n_rows):
        for col in range(n_cols):
            y = min(row * step, H - patch_size) if H >= patch_size else 0
            x = min(col * step, W - patch_size) if W >= patch_size else 0
            patch_w = min(patch_size, W - x)
            patch_h = min(patch_size, H - y)
            patch = image_pil.crop((x, y, x + patch_w, y + patch_h))
            if patch_w < patch_size or patch_h < patch_size:
                padded = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
                padded.paste(patch, (0, 0))
                patch = padded
            patches.append((patch, row, col, y, x, patch_h, patch_w))
    return patches, H, W, n_rows, n_cols


def list_tiff_files(input_dir: str) -> List[Path]:
    input_path = Path(input_dir)
    tiff_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    return tiff_files


def shard_indices(n: int, rank: int, world_size: int) -> range:
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return range(start, end)


# =============================================================================
# Distributed Setup  (UNCHANGED)
# =============================================================================
def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        if rank == 0:
            logger.info(f"Distributed initialized: world_size={world_size} ranks")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        logger.info("Running in single-GPU mode")
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Model Loading  (UNCHANGED)
# =============================================================================
def load_model(
    bpe_path: str,
    device: torch.device,
    finetuned_checkpoint: Optional[str] = None,
    original_checkpoint: Optional[str] = None,
    rank: int = 0,
) -> torch.nn.Module:
    if rank == 0:
        logger.info("Loading SAM3 model...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
    )
    if finetuned_checkpoint:
        if not original_checkpoint:
            raise ValueError("original_checkpoint required when using finetuned model")
        if rank == 0:
            logger.info(f"  Base checkpoint: {original_checkpoint}")
            logger.info(f"  Finetuned checkpoint: {finetuned_checkpoint}")
        original_state = load_state_dict_flexible(original_checkpoint)
        model.load_state_dict(original_state, strict=False)
        finetuned_state = load_state_dict_flexible(finetuned_checkpoint)
        missing, unexpected = model.load_state_dict(finetuned_state, strict=False)
        if rank == 0:
            if missing:
                logger.warning(f"  Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"  Unexpected keys: {len(unexpected)}")
            logger.info("✓ Finetuned model loaded")
    elif original_checkpoint:
        if rank == 0:
            logger.info(f"  Loading from checkpoint: {original_checkpoint}")
        original_state = load_state_dict_flexible(original_checkpoint)
        model.load_state_dict(original_state, strict=False)
        if rank == 0:
            logger.info("✓ Model loaded from checkpoint")
    else:
        if rank == 0:
            logger.info("✓ Pretrained model loaded")
    model.eval().to(device)
    return model


def create_transform(image_size: int = 1008) -> ComposeAPI:
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=image_size, max_size=image_size, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_postprocessor(confidence_threshold: float = 0.5) -> PostProcessImage:
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=confidence_threshold,
        to_cpu=False,
    )


# =============================================================================
# Core Inference  (UNCHANGED)
# =============================================================================
def process_patches_batch_native(
    patch_data_list: List[Tuple[int, Tuple]],
    prompts: List[str],
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    device: torch.device,
    patch_size: int,
) -> List[dict]:
    global GLOBAL_COUNTER
    datapoints: List[Datapoint] = []
    query_id_map: Dict[int, Tuple] = {}

    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_tuple
        dp = create_empty_datapoint()
        set_image(dp, patch_pil)
        for prompt in prompts:
            query_id = add_text_prompt(dp, prompt)
            query_id_map[query_id] = (
                batch_idx, prompt, global_idx, row, col, y_start, x_start, patch_h, patch_w,
            )
        dp = transform(dp)
        datapoints.append(dp)

    batch = collate(datapoints, dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    with torch.inference_mode():
        output = model(batch)

    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    batch_results: Dict[int, dict] = {}
    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_tuple
        batch_results[batch_idx] = {
            "idx": global_idx,
            "row": row, "col": col,
            "y_start": y_start, "x_start": x_start,
            "patch_h": patch_h, "patch_w": patch_w,
            "prompt_masks": {},
            "prompt_scores": {},
        }

    for query_id, (batch_idx, prompt, *_rest) in query_id_map.items():
        if query_id in processed_results:
            result = processed_results[query_id]
            if "masks" in result and len(result["masks"]) > 0:
                masks = result["masks"]
                scores = result.get("scores", None)
                if torch.is_tensor(masks):
                    masks = masks.cpu().numpy()
                if scores is not None and torch.is_tensor(scores):
                    scores = scores.cpu().numpy()
                batch_results[batch_idx]["prompt_masks"][prompt] = masks
                batch_results[batch_idx]["prompt_scores"][prompt] = scores
            else:
                batch_results[batch_idx]["prompt_masks"][prompt] = None
                batch_results[batch_idx]["prompt_scores"][prompt] = None
        else:
            batch_results[batch_idx]["prompt_masks"][prompt] = None
            batch_results[batch_idx]["prompt_scores"][prompt] = None

    return list(batch_results.values())


def process_patches_native_single_rank(
    patches: List[Tuple],
    prompts: List[str],
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    batch_size: int,
    device: torch.device,
    patch_size: int,
) -> List[dict]:
    results: List[dict] = []
    n = len(patches)
    for start in range(0, n, batch_size):
        chunk = patches[start: start + batch_size]
        patch_data_list = [(start + i, chunk[i]) for i in range(len(chunk))]
        batch_results = process_patches_batch_native(
            patch_data_list=patch_data_list,
            prompts=prompts, model=model,
            transform=transform, postprocessor=postprocessor,
            device=device, patch_size=patch_size,
        )
        results.extend(batch_results)
    return results


# =============================================================================
# Post-Processing: Mask-Level NMS  (REPLACES pixel-level stitching)
# =============================================================================
def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode bool (H,W) mask as RLE for memory-efficient storage."""
    flat = mask.flatten(order="F").astype(np.uint8)
    changes = np.where(np.diff(flat, prepend=flat[0] + 1, append=flat[-1] + 1))[0]
    counts = np.diff(changes).tolist()
    if flat[0] == 1:
        counts = [0] + counts
    return {"counts": counts, "size": list(mask.shape)}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Decode RLE back to bool (H,W) mask."""
    h, w = rle["size"]
    mask = np.zeros(h * w, dtype=np.uint8)
    idx, val = 0, 0
    for cnt in rle["counts"]:
        mask[idx: idx + cnt] = val
        idx += cnt
        val = 1 - val
    return mask.reshape(h, w, order="F").astype(bool)


def rle_intersection_area(rle_a: dict, rle_b: dict) -> int:
    return int((rle_to_mask(rle_a) & rle_to_mask(rle_b)).sum())


def collect_global_masks(
    results: List[dict],
    prompts: List[str],
    orig_h: int,
    orig_w: int,
    patch_size: int,
) -> List[dict]:
    """
    Project every detected mask from every patch into full-image coordinates.
    Stored as RLE to keep memory manageable.
    Returns list of { "rle", "score", "class", "area" }.
    """
    global_masks = []
    for result in results:
        y_start = result["y_start"]
        x_start = result["x_start"]
        patch_h  = result["patch_h"]
        patch_w  = result["patch_w"]

        for prompt in prompts:
            masks  = result["prompt_masks"].get(prompt)
            scores = result["prompt_scores"].get(prompt)

            if masks is None or (isinstance(masks, np.ndarray) and masks.size == 0):
                continue

            masks = np.array(masks)
            if scores is not None:
                scores = np.array(scores)
                if scores.size > 0 and (np.nanmax(scores) > 10 or np.nanmin(scores) < -10):
                    scores = 1.0 / (1.0 + np.exp(-scores))
            else:
                scores = np.ones(len(masks)) * 0.5

            for i, m in enumerate(masks):
                if m.ndim == 3:
                    m = m.squeeze(0)
                if m.shape != (patch_size, patch_size):
                    m_t = torch.from_numpy(m).float()[None, None]
                    m_t = torch.nn.functional.interpolate(
                        m_t, size=(patch_size, patch_size),
                        mode="bilinear", align_corners=False,
                    )
                    m = m_t.squeeze().numpy()

                m_crop = (m[:patch_h, :patch_w] > 0.5)
                area = int(m_crop.sum())
                if area == 0:
                    continue

                full_mask = np.zeros((orig_h, orig_w), dtype=bool)
                full_mask[y_start: y_start + patch_h,
                          x_start: x_start + patch_w] = m_crop

                global_masks.append({
                    "rle":   mask_to_rle(full_mask),
                    "score": float(scores[i]) if i < len(scores) else 0.5,
                    "class": prompt,
                    "area":  area,
                })

    return global_masks


def greedy_nms_masks(
    global_masks: List[dict],
    iou_threshold: float = 0.5,
    cross_class: bool = True,
) -> List[dict]:
    """
    Sort masks by descending confidence; accept a mask only if its IoU
    with every already-accepted mask is below iou_threshold.

    cross_class=True  → suppression across all classes (prevents same region
                        being assigned two different class ids)
    cross_class=False → each class NMS'd independently
    """
    if not global_masks:
        return []

    sorted_masks = sorted(global_masks, key=lambda x: x["score"], reverse=True)
    accepted: List[dict] = []

    for candidate in sorted_masks:
        suppress = False
        for kept in accepted:
            if not cross_class and kept["class"] != candidate["class"]:
                continue
            inter = rle_intersection_area(candidate["rle"], kept["rle"])
            if inter == 0:
                continue
            union = candidate["area"] + kept["area"] - inter
            if inter / max(union, 1) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            accepted.append(candidate)

    logger.debug(f"NMS: {len(sorted_masks)} → {len(accepted)} masks kept")
    return accepted


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    Image.fromarray(mask_uint8).save(output_path, compression="tiff_deflate")


def rasterize_per_class(
    accepted: List[dict], prompts: List[str], orig_h: int, orig_w: int
) -> Dict[str, np.ndarray]:
    """Binary float32 canvas per class."""
    canvases = {p: np.zeros((orig_h, orig_w), dtype=np.float32) for p in prompts}
    for m in accepted:
        if m["class"] in canvases:
            canvases[m["class"]] = np.maximum(
                canvases[m["class"]], rle_to_mask(m["rle"]).astype(np.float32)
            )
    return canvases


def rasterize_combined(
    accepted: List[dict], class_colors: Dict[str, int], orig_h: int, orig_w: int
) -> np.ndarray:
    """Label map (uint8): highest-scoring class wins per pixel."""
    combined  = np.zeros((orig_h, orig_w), dtype=np.uint8)
    score_map = np.full((orig_h, orig_w), -1.0, dtype=np.float32)
    for m in accepted:
        decoded = rle_to_mask(m["rle"])
        update  = decoded & (m["score"] > score_map)
        combined[update]  = class_colors.get(m["class"], 0)
        score_map[update] = m["score"]
    return combined


def rasterize_overlay(
    accepted: List[dict],
    orig_h: int, orig_w: int,
    image_path: Path,
    alpha: float = 0.45,
) -> np.ndarray:
    """Colour overlay blended with original image."""
    img = Image.open(image_path)
    if img.mode not in ["RGB", "L", "RGBA"]:
        if img.mode in ["I", "I;16", "F"]:
            arr = np.array(img).astype(np.float32)
            lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
            arr = np.clip(arr, lo, hi)
            arr = ((arr - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            img = img.convert("L")
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_arr = np.array(img)

    overlay   = np.zeros((orig_h, orig_w, 4), dtype=np.float32)
    score_map = np.full((orig_h, orig_w), -1.0, dtype=np.float32)

    for m in accepted:
        color = CLASS_COLORS_RGB.get(m["class"])
        if color is None:
            continue
        decoded = rle_to_mask(m["rle"])
        update  = decoded & (m["score"] > score_map)
        overlay[update, 0] = color[0]
        overlay[update, 1] = color[1]
        overlay[update, 2] = color[2]
        overlay[update, 3] = alpha
        score_map[update]  = m["score"]

    a = overlay[:, :, 3:4]
    return (img_arr * (1 - a) + overlay[:, :, :3] * 255 * a).astype(np.uint8)


# =============================================================================
# End-to-End Single-Image Processing  (CHANGED: post-inference block only)
# =============================================================================
def process_and_save_one_image_single_rank(
    image_path: Path,
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    prompts: List[str],
    class_colors: Dict[str, int],
    output_dir: Path,
    patch_size: int,
    overlap_ratio: float,
    batch_size: int,
    device: torch.device,
    rank: int,
    nms_iou_threshold: float = 0.5,   # NEW
    cross_class_nms: bool = True,      # NEW
) -> float:
    global GLOBAL_COUNTER
    GLOBAL_COUNTER = 1

    t0 = time.time()
    image_pil  = tif_to_rgb_pil(str(image_path))
    image_name = image_path.name

    patches, orig_h, orig_w, n_rows, n_cols = crop_image_to_patches(
        image_pil, patch_size, overlap_ratio
    )
    logger.info(
        f"[rank {rank}] Processing {image_name} | size={orig_w}x{orig_h} | "
        f"patches={len(patches)} ({n_rows}x{n_cols}) | overlap={overlap_ratio*100:.0f}%"
    )

    # --- inference (UNCHANGED) ---
    local_results = process_patches_native_single_rank(
        patches=patches, prompts=prompts, model=model,
        transform=transform, postprocessor=postprocessor,
        batch_size=batch_size, device=device, patch_size=patch_size,
    )

    # --- mask-level NMS (REPLACES pixel-level stitching) ---
    global_masks = collect_global_masks(local_results, prompts, orig_h, orig_w, patch_size)
    logger.info(f"[rank {rank}] {image_name} | {len(global_masks)} raw masks collected")

    accepted = greedy_nms_masks(global_masks,
                                iou_threshold=nms_iou_threshold,
                                cross_class=cross_class_nms)
    logger.info(f"[rank {rank}] {image_name} | {len(accepted)} masks after NMS")

    image_stem = image_path.stem

    # per-class binary masks
    for prompt, canvas in rasterize_per_class(accepted, prompts, orig_h, orig_w).items():
        save_mask(canvas, output_dir / prompt.replace(" ", "_") / f"{image_stem}_mask.tif")

    # combined label map
    combined      = rasterize_combined(accepted, class_colors, orig_h, orig_w)
    combined_path = output_dir / "combined" / f"{image_stem}_combined.tif"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(combined_path, compression="tiff_deflate")

    # overlay
    overlay_arr  = rasterize_overlay(accepted, orig_h, orig_w, image_path, alpha=0.45)
    overlay_path = output_dir / "combined_overlay" / f"{image_stem}_overlay.tif"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_arr).save(overlay_path, compression="tiff_deflate")

    dt = time.time() - t0
    logger.info(f"[rank {rank}] ✓ Saved masks for {image_name} in {dt:.1f}s")
    return dt


# =============================================================================
# Main Entry Point  (CHANGED: two new CLI args + pass-through)
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 Inference with Native Batching + Mask-Level NMS (IMAGE-SHARDED)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir",             type=str,   required=True)
    parser.add_argument("--bpe-path",              type=str,   required=True)
    parser.add_argument("--output-dir",            type=str,   default="./output")
    parser.add_argument("--finetuned-checkpoint",  type=str,   default=None)
    parser.add_argument("--original-checkpoint",   type=str,   default=None)
    parser.add_argument("--patch-size",            type=int,   default=512)
    parser.add_argument("--overlap-ratio",         type=float, default=0.25)
    parser.add_argument("--batch-size",            type=int,   default=8)
    parser.add_argument("--confidence",  nargs="+", type=float, default=[0.5])
    parser.add_argument("--image-size",            type=int,   default=1008)
    parser.add_argument("--prompts", nargs="+",
        default=["cortex", "Phloem Fibers", "Xylem vessels", "Pith cells", "outer cells"])
    parser.add_argument("--skip-existing", action="store_true")
    # NEW
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5,
        help="IoU threshold for mask-level NMS (lower = more aggressive suppression)")
    parser.add_argument("--no-cross-class-nms", action="store_true",
        help="Disable cross-class NMS (each class NMS'd independently)")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.confidence) == 1:
        confidence_thresholds = args.confidence * len(args.prompts)
    elif len(args.confidence) == len(args.prompts):
        confidence_thresholds = args.confidence
    else:
        raise ValueError(
            f"Confidence list length ({len(args.confidence)}) must be 1 "
            f"or match number of prompts ({len(args.prompts)})"
        )

    all_files = list_tiff_files(args.input_dir)

    if args.skip_existing:
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        existing  = {p.stem.replace("_combined", "") for p in combined_dir.glob("*.tif")}
        all_files = [p for p in all_files if p.stem not in existing]

    if rank == 0:
        logger.info(f"{'='*70}")
        logger.info("Starting distributed inference (IMAGE-SHARDED + MASK-LEVEL NMS)")
        logger.info(f"{'='*70}")
        logger.info(f"  world_size:        {world_size}")
        logger.info(f"  input dir:         {args.input_dir}")
        logger.info(f"  output dir:        {output_dir}")
        logger.info(f"  images total:      {len(all_files)}")
        logger.info(f"  prompts:           {args.prompts}")
        logger.info(f"  confidence:        {confidence_thresholds}")
        logger.info(f"  patch_size:        {args.patch_size} | overlap: {args.overlap_ratio*100:.0f}% | batch_size: {args.batch_size}")
        logger.info(f"  NMS IoU threshold: {args.nms_iou_threshold}")
        logger.info(f"  cross-class NMS:   {not args.no_cross_class_nms}")
        logger.info(f"{'='*70}")

    if not all_files:
        if rank == 0:
            logger.info("No images to process. Exiting.")
        cleanup_distributed()
        return

    my_range = shard_indices(len(all_files), rank, world_size)
    my_files = [all_files[i] for i in my_range]
    logger.info(f"[rank {rank}] Assigned {len(my_files)} / {len(all_files)} images")

    model = load_model(
        bpe_path=args.bpe_path, device=device,
        finetuned_checkpoint=args.finetuned_checkpoint,
        original_checkpoint=args.original_checkpoint,
        rank=rank,
    )
    transform    = create_transform(args.image_size)
    postprocessor = create_postprocessor(args.confidence[0])
    class_colors = {prompt: idx + 1 for idx, prompt in enumerate(args.prompts)}

    t_total0 = time.time()
    per_image_times: List[float] = []
    processed_count = 0

    for img_path in my_files:
        dt = process_and_save_one_image_single_rank(
            image_path=img_path,
            model=model, transform=transform, postprocessor=postprocessor,
            prompts=args.prompts, class_colors=class_colors,
            output_dir=output_dir,
            patch_size=args.patch_size, overlap_ratio=args.overlap_ratio,
            batch_size=args.batch_size, device=device, rank=rank,
            nms_iou_threshold=args.nms_iou_threshold,       # NEW
            cross_class_nms=not args.no_cross_class_nms,    # NEW
        )
        per_image_times.append(dt)
        processed_count += 1

    my_total = time.time() - t_total0

    if world_size > 1 and dist.is_initialized():
        t_count = torch.tensor([processed_count], device=device, dtype=torch.long)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        t_max = torch.tensor([my_total], device=device, dtype=torch.float64)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        if rank == 0:
            total_images = int(t_count.item())
            max_time     = float(t_max.item())
            img_per_min  = total_images / (max_time / 60.0) if max_time > 0 else 0.0
            logger.info(f"{'='*70}")
            logger.info("Inference Complete (IMAGE-SHARDED + MASK-LEVEL NMS)")
            logger.info(f"  total images: {total_images}")
            logger.info(f"  makespan:     {max_time:.1f}s ({max_time/60:.1f} min)")
            logger.info(f"  throughput:   {img_per_min:.2f} images/min")
            logger.info(f"  output:       {output_dir}")
            logger.info(f"{'='*70}")
    else:
        if rank == 0:
            img_per_min = processed_count / (my_total / 60.0) if my_total > 0 else 0.0
            logger.info(f"{'='*70}")
            logger.info("Inference Complete (single rank + MASK-LEVEL NMS)")
            logger.info(f"  images:     {processed_count}")
            logger.info(f"  total time: {my_total:.1f}s ({my_total/60:.1f} min)")
            logger.info(f"  throughput: {img_per_min:.2f} images/min")
            logger.info(f"  output:     {output_dir}")
            logger.info(f"{'='*70}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
