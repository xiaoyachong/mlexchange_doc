"""
SAM3 Distributed Segmentation Inference - Native Batching Version (IMAGE-SHARDED)
WITH GLOBAL INSTANCE NMS + CLEAN RASTERIZATION

Key changes vs v6:
✅ 25% overlap in patching (unchanged)
✅ Global instance NMS replaces pixel-wise max-confidence stitching
✅ Masks stored as crop+offset during NMS (no full-image alloc until rasterization)
✅ Clean rasterize_per_class / rasterize_combined / rasterize_overlay
✅ SHARD BY IMAGE ACROSS RANKS (unchanged)

Author: ALS Photon Science Computing
"""

import os
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
    "Cortex":                    (0.0,        0.0,        1.0),
    "Phloem Fibers":             (0.0,        1.0,        0.0),
    "Phloem":                    (128/255,    0.0,        128/255),
    "Hydrated Xylem vessels":    (1.0,        0.0,        0.0),
    "Air-based Pith cells":      (1.0,        1.0,        0.0),
    "Water-based Pith cells":    (1.0,        165/255,    0.0),
    "Dehydrated Xylem vessels":  (0.0,        199/255,    190/255),
}


# =============================================================================
# SAM3 Native API Helpers  (UNCHANGED)
# =============================================================================
def create_empty_datapoint() -> Datapoint:
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint: Datapoint, pil_image: Image.Image) -> None:
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint: Datapoint, text_query: str) -> int:
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1
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
# Image / IO Utilities  (UNCHANGED)
# =============================================================================
def load_state_dict_flexible(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def tif_to_rgb_pil(image_path: str, plo: float = 0.5, phi: float = 99.5) -> Image.Image:
    img = Image.open(image_path)
    if img.mode not in ["RGB", "L", "RGBA"]:
        if img.mode in ["I", "I;16", "F"]:
            arr = np.array(img).astype(np.float32)
            lo, hi = np.percentile(arr, plo), np.percentile(arr, phi)
            arr = np.clip(arr, lo, hi)
            arr = ((arr - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
            img = Image.fromarray(arr)
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
    step = int(patch_size * (1 - overlap_ratio))
    n_rows = max(1, (H - patch_size) // step + 1)
    n_cols = max(1, (W - patch_size) // step + 1)
    if n_rows > 1 and (n_rows - 1) * step + patch_size < H:
        n_rows += 1
    if n_cols > 1 and (n_cols - 1) * step + patch_size < W:
        n_cols += 1
    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            y = min(row * step, H - patch_size) if H >= patch_size else 0
            x = min(col * step, W - patch_size) if W >= patch_size else 0
            ph, pw = min(patch_size, H - y), min(patch_size, W - x)
            patch = image_pil.crop((x, y, x + pw, y + ph))
            if pw < patch_size or ph < patch_size:
                padded = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
                padded.paste(patch, (0, 0))
                patch = padded
            patches.append((patch, row, col, y, x, ph, pw))
    return patches, H, W, n_rows, n_cols


def list_tiff_files(input_dir: str) -> List[Path]:
    p = Path(input_dir)
    files = sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
    if not files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    return files


def shard_indices(n: int, rank: int, world_size: int) -> range:
    base, rem = n // world_size, n % world_size
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
            logger.info(f"Distributed initialized: world_size={world_size}")
    else:
        rank, world_size, local_rank = 0, 1, 0
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
        bpe_path=bpe_path, device=device, eval_mode=True, enable_segmentation=True,
    )
    if finetuned_checkpoint:
        if not original_checkpoint:
            raise ValueError("original_checkpoint required when using finetuned model")
        if rank == 0:
            logger.info(f"  Base: {original_checkpoint}  Finetuned: {finetuned_checkpoint}")
        model.load_state_dict(load_state_dict_flexible(original_checkpoint), strict=False)
        missing, unexpected = model.load_state_dict(
            load_state_dict_flexible(finetuned_checkpoint), strict=False
        )
        if rank == 0:
            if missing:    logger.warning(f"  Missing keys: {len(missing)}")
            if unexpected: logger.warning(f"  Unexpected keys: {len(unexpected)}")
            logger.info("✓ Finetuned model loaded")
    elif original_checkpoint:
        if rank == 0:
            logger.info(f"  Checkpoint: {original_checkpoint}")
        model.load_state_dict(load_state_dict_flexible(original_checkpoint), strict=False)
        if rank == 0:
            logger.info("✓ Model loaded")
    else:
        if rank == 0:
            logger.info("✓ Pretrained model loaded")
    model.eval().to(device)
    return model


def create_transform(image_size: int = 1008) -> ComposeAPI:
    return ComposeAPI(transforms=[
        RandomResizeAPI(sizes=image_size, max_size=image_size, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_postprocessor(confidence_threshold: float = 0.5) -> PostProcessImage:
    return PostProcessImage(
        max_dets_per_img=-1, iou_type="segm",
        use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=confidence_threshold, to_cpu=False,
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
    datapoints, query_id_map = [], {}

    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_tuple
        dp = create_empty_datapoint()
        set_image(dp, patch_pil)
        for prompt in prompts:
            qid = add_text_prompt(dp, prompt)
            query_id_map[qid] = (batch_idx, prompt, global_idx, row, col, y_start, x_start, patch_h, patch_w)
        dp = transform(dp)
        datapoints.append(dp)

    batch = collate(datapoints, dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    with torch.inference_mode():
        output = model(batch)
    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    batch_results: Dict[int, dict] = {}
    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        _, row, col, y_start, x_start, patch_h, patch_w = patch_tuple
        batch_results[batch_idx] = {
            "idx": global_idx, "row": row, "col": col,
            "y_start": y_start, "x_start": x_start,
            "patch_h": patch_h, "patch_w": patch_w,
            "prompt_masks": {}, "prompt_scores": {},
        }

    for qid, (batch_idx, prompt, *_) in query_id_map.items():
        if qid in processed_results:
            res = processed_results[qid]
            if "masks" in res and len(res["masks"]) > 0:
                masks  = res["masks"]
                scores = res.get("scores", None)
                if torch.is_tensor(masks):  masks  = masks.cpu().numpy()
                if scores is not None and torch.is_tensor(scores): scores = scores.cpu().numpy()
                batch_results[batch_idx]["prompt_masks"][prompt]  = masks
                batch_results[batch_idx]["prompt_scores"][prompt] = scores
            else:
                batch_results[batch_idx]["prompt_masks"][prompt]  = None
                batch_results[batch_idx]["prompt_scores"][prompt] = None
        else:
            batch_results[batch_idx]["prompt_masks"][prompt]  = None
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
    results = []
    for start in range(0, len(patches), batch_size):
        chunk = patches[start : start + batch_size]
        results.extend(process_patches_batch_native(
            patch_data_list=[(start + i, chunk[i]) for i in range(len(chunk))],
            prompts=prompts, model=model, transform=transform,
            postprocessor=postprocessor, device=device, patch_size=patch_size,
        ))
    return results


# =============================================================================
# NEW: Instance collection — crop+offset only, no full-image alloc
# =============================================================================
def _decode_mask_crop(m: np.ndarray, patch_h: int, patch_w: int, patch_size: int) -> np.ndarray:
    """Resize to patch_size if needed, crop to (patch_h, patch_w), return bool array."""
    if m.ndim == 3:
        m = m.squeeze(0)
    if m.shape != (patch_size, patch_size):
        t = torch.from_numpy(m).float()[None, None]
        t = torch.nn.functional.interpolate(
            t, size=(patch_size, patch_size), mode="bilinear", align_corners=False
        )
        m = t.squeeze().numpy()
    return (m[:patch_h, :patch_w] > 0.5)


def collect_instances(
    results: List[dict],
    prompts: List[str],
    patch_size: int,
) -> List[dict]:
    """
    Gather every detected mask instance across all patches and prompts.
    Masks are kept as small (patch_h x patch_w) bool crops + their image-space
    offsets.  No full (H x W) array is allocated here.

    Each entry: {mask_crop, y_start, x_start, patch_h, patch_w, score, class, area}
    """
    instances = []
    for r in results:
        y_start, x_start = r["y_start"], r["x_start"]
        ph, pw = r["patch_h"], r["patch_w"]

        for prompt in prompts:
            masks  = r["prompt_masks"].get(prompt)
            scores = r["prompt_scores"].get(prompt)
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
                crop = _decode_mask_crop(m, ph, pw, patch_size)
                area = int(crop.sum())
                if area == 0:
                    continue
                instances.append({
                    "mask_crop": crop,          # bool (ph x pw) — small
                    "y_start":   y_start,
                    "x_start":   x_start,
                    "patch_h":   ph,
                    "patch_w":   pw,
                    "score":     float(scores[i]) if i < len(scores) else 0.5,
                    "class":     prompt,
                    "area":      area,
                })
    return instances


# =============================================================================
# NEW: Greedy NMS — intersection-region IoU, no full-image decode
# =============================================================================
def greedy_nms(
    instances: List[dict],
    iou_threshold: float = 0.5,
    cross_class: bool = True,
) -> List[dict]:
    """
    Sort by descending confidence; keep a candidate only if its IoU with every
    already-kept mask is below iou_threshold.

    IoU is computed over the intersection bounding box of the two patch regions,
    so we never need to allocate full-image arrays during NMS.

    cross_class=True  → suppression across all classes (prevents double-labelling
                        the same region with two different class ids)
    cross_class=False → each class is NMS'd independently
    """
    if not instances:
        return []

    sorted_inst = sorted(instances, key=lambda x: x["score"], reverse=True)
    kept: List[dict] = []

    for cand in sorted_inst:
        suppress = False
        cy0 = cand["y_start"];  cy1 = cy0 + cand["patch_h"]
        cx0 = cand["x_start"];  cx1 = cx0 + cand["patch_w"]

        for acc in kept:
            if not cross_class and acc["class"] != cand["class"]:
                continue

            # Fast reject: patches don't overlap → cannot be the same object
            ay0 = acc["y_start"];  ay1 = ay0 + acc["patch_h"]
            ax0 = acc["x_start"];  ax1 = ax0 + acc["patch_w"]
            if cy1 <= ay0 or ay1 <= cy0 or cx1 <= ax0 or ax1 <= cx0:
                continue

            # Intersection region in image coordinates
            iy0, iy1 = max(cy0, ay0), min(cy1, ay1)
            ix0, ix1 = max(cx0, ax0), min(cx1, ax1)

            # Slice each crop to the intersection region
            c_region = cand["mask_crop"][iy0 - cy0 : iy1 - cy0, ix0 - cx0 : ix1 - cx0]
            a_region = acc["mask_crop"] [iy0 - ay0 : iy1 - ay0, ix0 - ax0 : ix1 - ax0]

            inter = int((c_region & a_region).sum())
            if inter == 0:
                continue
            union = cand["area"] + acc["area"] - inter
            if inter / max(union, 1) >= iou_threshold:
                suppress = True
                break

        if not suppress:
            kept.append(cand)

    logger.debug(f"NMS: {len(sorted_inst)} → {len(kept)} instances kept")
    return kept


# =============================================================================
# NEW: Rasterization — full-image arrays allocated only here, once per output
# =============================================================================
def rasterize_per_class(
    kept: List[dict], prompts: List[str], orig_h: int, orig_w: int
) -> Dict[str, np.ndarray]:
    """Binary float32 canvas per class. One full-image array per class."""
    canvases = {p: np.zeros((orig_h, orig_w), dtype=np.float32) for p in prompts}
    for m in kept:
        cls = m["class"]
        if cls not in canvases:
            continue
        y0, x0 = m["y_start"], m["x_start"]
        y1, x1 = y0 + m["patch_h"], x0 + m["patch_w"]
        # After NMS, same-class overlap is already resolved; paint score directly
        canvases[cls][y0:y1, x0:x1][m["mask_crop"]] = m["score"]
    return canvases


def rasterize_combined(
    kept: List[dict], class_colors: Dict[str, int], orig_h: int, orig_w: int
) -> np.ndarray:
    """Label map (uint8): highest-scoring class wins per pixel."""
    combined  = np.zeros((orig_h, orig_w), dtype=np.uint8)
    score_map = np.full((orig_h, orig_w), -1.0, dtype=np.float32)
    for m in kept:
        y0, x0 = m["y_start"], m["x_start"]
        y1, x1 = y0 + m["patch_h"], x0 + m["patch_w"]
        update = m["mask_crop"] & (m["score"] > score_map[y0:y1, x0:x1])
        combined [y0:y1, x0:x1][update] = class_colors.get(m["class"], 0)
        score_map[y0:y1, x0:x1][update] = m["score"]
    return combined


def rasterize_overlay(
    kept: List[dict],
    orig_h: int,
    orig_w: int,
    image_path: Path,
    alpha: float = 0.45,
) -> np.ndarray:
    """Colour overlay blended with the original image."""
    img = tif_to_rgb_pil(str(image_path))
    img_arr = np.array(img)

    overlay   = np.zeros((orig_h, orig_w, 4), dtype=np.float32)
    score_map = np.full((orig_h, orig_w), -1.0, dtype=np.float32)

    for m in kept:
        color = CLASS_COLORS_RGB.get(m["class"])
        if color is None:
            continue
        y0, x0 = m["y_start"], m["x_start"]
        y1, x1 = y0 + m["patch_h"], x0 + m["patch_w"]
        update = m["mask_crop"] & (m["score"] > score_map[y0:y1, x0:x1])
        overlay[y0:y1, x0:x1][update, 0] = color[0]
        overlay[y0:y1, x0:x1][update, 1] = color[1]
        overlay[y0:y1, x0:x1][update, 2] = color[2]
        overlay[y0:y1, x0:x1][update, 3] = alpha
        score_map[y0:y1, x0:x1][update]  = m["score"]

    a = overlay[:, :, 3:4]
    return (img_arr * (1 - a) + overlay[:, :, :3] * 255 * a).astype(np.uint8)


# =============================================================================
# Save helpers
# =============================================================================
def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    Image.fromarray(mask_u8).save(output_path, compression="tiff_deflate")


def save_array(arr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(output_path, compression="tiff_deflate")


# =============================================================================
# Main per-image pipeline
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
    iou_threshold: float = 0.5,
    cross_class_nms: bool = True,
) -> float:
    global GLOBAL_COUNTER
    GLOBAL_COUNTER = 1

    t0 = time.time()
    image_pil  = tif_to_rgb_pil(str(image_path))
    image_name = image_path.name
    image_stem = image_path.stem

    patches, orig_h, orig_w, n_rows, n_cols = crop_image_to_patches(
        image_pil, patch_size, overlap_ratio
    )
    logger.info(
        f"[rank {rank}] {image_name} | {orig_w}x{orig_h} | "
        f"{len(patches)} patches ({n_rows}x{n_cols}) | overlap={overlap_ratio*100:.0f}%"
    )

    # ── Inference ────────────────────────────────────────────────────────────
    raw_results = process_patches_native_single_rank(
        patches=patches, prompts=prompts, model=model, transform=transform,
        postprocessor=postprocessor, batch_size=batch_size,
        device=device, patch_size=patch_size,
    )

    # ── Collect instances (crop+offset, no full-image alloc) ─────────────────
    instances = collect_instances(raw_results, prompts, patch_size)
    logger.info(f"[rank {rank}] {image_name} | {len(instances)} instances before NMS")

    # ── Global NMS ───────────────────────────────────────────────────────────
    kept = greedy_nms(instances, iou_threshold=iou_threshold, cross_class=cross_class_nms)
    logger.info(f"[rank {rank}] {image_name} | {len(kept)} instances after NMS")

    # ── Rasterize & save ─────────────────────────────────────────────────────
    # Per-class binary masks
    per_class = rasterize_per_class(kept, prompts, orig_h, orig_w)
    for prompt, canvas in per_class.items():
        prompt_dir = output_dir / prompt.replace(" ", "_")
        save_mask(canvas, prompt_dir / f"{image_stem}_mask.tif")

    # Combined label map
    combined = rasterize_combined(kept, class_colors, orig_h, orig_w)
    save_array(combined, output_dir / "combined" / f"{image_stem}_combined.tif")

    # Overlay
    overlay = rasterize_overlay(kept, orig_h, orig_w, image_path, alpha=0.45)
    save_array(overlay, output_dir / "combined_overlay" / f"{image_stem}_overlay.tif")

    dt = time.time() - t0
    logger.info(f"[rank {rank}] ✓ {image_name} done in {dt:.1f}s")
    return dt


# =============================================================================
# Main Entry Point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 Inference — Global Instance NMS + Clean Rasterization",
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
    parser.add_argument("--image-size",            type=int,   default=1008)
    parser.add_argument("--iou-threshold",         type=float, default=0.5,
                        help="IoU threshold for greedy NMS.")
    parser.add_argument("--no-cross-class-nms",    action="store_true",
                        help="If set, NMS runs per-class instead of across all classes.")
    parser.add_argument(
        "--confidence", nargs="+", type=float, default=[0.5],
        help="Confidence threshold(s) — one value or one per prompt.",
    )
    parser.add_argument(
        "--prompts", nargs="+",
        default=[
            "Cortex", "Phloem Fibers", "Phloem",
            "Hydrated Xylem vessels", "Air-based Pith cells",
            "Water-based Pith cells", "Dehydrated Xylem vessels",
        ],
    )
    parser.add_argument("--skip-existing", action="store_true")
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
            f"--confidence must have 1 value or one per prompt ({len(args.prompts)})"
        )

    all_files = list_tiff_files(args.input_dir)

    if args.skip_existing:
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem.replace("_combined", "") for p in combined_dir.glob("*.tif")}
        all_files = [f for f in all_files if f.stem not in existing]

    if rank == 0:
        logger.info(f"{'='*70}")
        logger.info("SAM3 Inference — Global Instance NMS + Clean Rasterization")
        logger.info(f"{'='*70}")
        logger.info(f"  world_size:            {world_size}")
        logger.info(f"  input dir:             {args.input_dir}")
        logger.info(f"  output dir:            {output_dir}")
        logger.info(f"  images (after skip):   {len(all_files)}")
        logger.info(f"  prompts:               {args.prompts}")
        logger.info(f"  confidence thresholds: {confidence_thresholds}")
        logger.info(f"  patch_size:            {args.patch_size}")
        logger.info(f"  overlap:               {args.overlap_ratio*100:.0f}%")
        logger.info(f"  batch_size:            {args.batch_size}")
        logger.info(f"  iou_threshold:         {args.iou_threshold}")
        logger.info(f"  cross_class_nms:       {not args.no_cross_class_nms}")
        logger.info(f"{'='*70}")

    if not all_files:
        if rank == 0:
            logger.info("No images to process. Exiting.")
        cleanup_distributed()
        return

    my_files = [all_files[i] for i in shard_indices(len(all_files), rank, world_size)]
    logger.info(f"[rank {rank}] Assigned {len(my_files)} / {len(all_files)} images")

    model        = load_model(args.bpe_path, device, args.finetuned_checkpoint,
                              args.original_checkpoint, rank)
    transform    = create_transform(args.image_size)
    postprocessor= create_postprocessor(args.confidence[0])
    class_colors = {p: idx + 1 for idx, p in enumerate(args.prompts)}

    t_total0 = time.time()
    per_image_times, processed_count = [], 0

    for img_path in my_files:
        dt = process_and_save_one_image_single_rank(
            image_path=img_path,
            model=model, transform=transform, postprocessor=postprocessor,
            prompts=args.prompts, class_colors=class_colors,
            output_dir=output_dir,
            patch_size=args.patch_size, overlap_ratio=args.overlap_ratio,
            batch_size=args.batch_size, device=device, rank=rank,
            iou_threshold=args.iou_threshold,
            cross_class_nms=not args.no_cross_class_nms,
        )
        per_image_times.append(dt)
        processed_count += 1

    my_total = time.time() - t_total0

    if world_size > 1 and dist.is_initialized():
        t_count = torch.tensor([processed_count], device=device, dtype=torch.long)
        t_max   = torch.tensor([my_total],        device=device, dtype=torch.float64)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_max,   op=dist.ReduceOp.MAX)
        if rank == 0:
            n, t = int(t_count.item()), float(t_max.item())
            logger.info(f"{'='*70}")
            logger.info(f"  total images:  {n}")
            logger.info(f"  makespan:      {t:.1f}s  ({t/60:.1f} min)")
            logger.info(f"  throughput:    {n/(t/60):.2f} images/min")
            logger.info(f"  output:        {output_dir}")
            logger.info(f"{'='*70}")
    else:
        if rank == 0:
            t = my_total
            logger.info(f"{'='*70}")
            logger.info(f"  images processed: {processed_count}")
            logger.info(f"  total time:       {t:.1f}s  ({t/60:.1f} min)")
            logger.info(f"  throughput:       {processed_count/(t/60):.2f} images/min")
            logger.info(f"  output:           {output_dir}")
            logger.info(f"{'='*70}")

    cleanup_distributed()


if __name__ == "__main__":
    main()