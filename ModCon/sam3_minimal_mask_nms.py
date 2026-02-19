"""
MINIMAL CHANGE: Replace pixel-level stitching with mask-level NMS.
Only the post-processing section changes. Everything else is identical to the original.

CHANGES:
  - REMOVED: stitch_masks_for_prompt_max_confidence()
  - REMOVED: save_combined_mask_max_confidence()
  - REMOVED: save_overlay() (pixel-level version)
  - ADDED:   collect_global_masks()       -- project patch masks → full image coords (RLE)
  - ADDED:   greedy_nms_masks()           -- deduplicate at mask level
  - ADDED:   rasterize_*()               -- paint accepted masks to output arrays
  - CHANGED: process_and_save_one_image_single_rank() -- calls new pipeline
  - CHANGED: argparse -- adds --nms-iou-threshold and --no-cross-class-nms
"""

# ============================================================
# (All original imports, env setup, logging, CLASS_COLORS_RGB,
#  SAM3 helpers, image utilities, distributed setup, model
#  loading, create_transform, create_postprocessor,
#  process_patches_batch_native, process_patches_native_single_rank
#  are UNCHANGED — keep them exactly as in your original file.)
# ============================================================


# =============================================================================
# RLE helpers (NEW — needed for sparse mask storage)
# =============================================================================
def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a bool (H,W) mask as RLE for memory-efficient storage."""
    flat = mask.flatten(order="F").astype(np.uint8)
    changes = np.where(np.diff(flat, prepend=flat[0] + 1, append=flat[-1] + 1))[0]
    counts = np.diff(changes).tolist()
    if flat[0] == 1:          # mask starts with foreground → prepend empty background run
        counts = [0] + counts
    return {"counts": counts, "size": list(mask.shape)}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Decode RLE back to a bool (H,W) mask."""
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


# =============================================================================
# REPLACES: stitch_masks_for_prompt_max_confidence
#           save_combined_mask_max_confidence
#           save_overlay
# =============================================================================
def collect_global_masks(
    results: List[dict],
    prompts: List[str],
    orig_h: int,
    orig_w: int,
    patch_size: int,
) -> List[dict]:
    """
    Project every detected mask from every patch into full-image coordinates.
    Store as RLE (sparse) to keep memory manageable.

    Returns a flat list of:
        { "rle": ..., "score": float, "class": str, "area": int }
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
                    scores = 1.0 / (1.0 + np.exp(-scores))   # sigmoid normalise
            else:
                scores = np.ones(len(masks)) * 0.5

            for i, m in enumerate(masks):
                if m.ndim == 3:
                    m = m.squeeze(0)

                # resize to patch_size if needed
                if m.shape != (patch_size, patch_size):
                    m_t = torch.from_numpy(m).float()[None, None]
                    m_t = torch.nn.functional.interpolate(
                        m_t, size=(patch_size, patch_size),
                        mode="bilinear", align_corners=False,
                    )
                    m = m_t.squeeze().numpy()

                m_crop = (m[:patch_h, :patch_w] > 0.5)
                area   = int(m_crop.sum())
                if area == 0:
                    continue

                # place into full-image canvas
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
    Sort by descending score; accept a mask only if IoU with every
    already-accepted mask is below iou_threshold.

    cross_class=True  → suppression works across all classes
                        (prevents two classes claiming the same region)
    cross_class=False → each class is NMS'd independently
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

    return accepted


def rasterize_per_class(
    accepted: List[dict], prompts: List[str], orig_h: int, orig_w: int
) -> Dict[str, np.ndarray]:
    """Binary float32 canvas per class from accepted masks."""
    canvases = {p: np.zeros((orig_h, orig_w), dtype=np.float32) for p in prompts}
    for m in accepted:
        if m["class"] in canvases:
            decoded = rle_to_mask(m["rle"]).astype(np.float32)
            canvases[m["class"]] = np.maximum(canvases[m["class"]], decoded)
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
    """Colour overlay blended with the original image."""
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

    a      = overlay[:, :, 3:4]
    result = img_arr * (1 - a) + overlay[:, :, :3] * 255 * a
    return result.astype(np.uint8)


# =============================================================================
# CHANGED: process_and_save_one_image_single_rank
# Only the post-inference block changes; signature gains two new kwargs.
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
    # NEW
    nms_iou_threshold: float = 0.5,
    cross_class_nms: bool = True,
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
        f"[rank {rank}] {image_name} | {orig_w}x{orig_h} | "
        f"{len(patches)} patches ({n_rows}x{n_cols}) | overlap={overlap_ratio*100:.0f}%"
    )

    # --- inference (UNCHANGED) ---
    local_results = process_patches_native_single_rank(
        patches=patches, prompts=prompts, model=model,
        transform=transform, postprocessor=postprocessor,
        batch_size=batch_size, device=device, patch_size=patch_size,
    )

    # --- NEW: mask-level NMS pipeline ---
    global_masks = collect_global_masks(local_results, prompts, orig_h, orig_w, patch_size)
    logger.info(f"[rank {rank}] {image_name} | {len(global_masks)} raw masks collected")

    accepted = greedy_nms_masks(global_masks, iou_threshold=nms_iou_threshold,
                                cross_class=cross_class_nms)
    logger.info(f"[rank {rank}] {image_name} | {len(accepted)} masks after NMS")

    image_stem = image_path.stem

    # per-class masks
    class_canvases = rasterize_per_class(accepted, prompts, orig_h, orig_w)
    for prompt, canvas in class_canvases.items():
        prompt_dir = output_dir / prompt.replace(" ", "_")
        save_mask(canvas, prompt_dir / f"{image_stem}_mask.tif")

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
    logger.info(f"[rank {rank}] ✓ {image_name} done in {dt:.1f}s")
    return dt


# =============================================================================
# CHANGED: main() — add two new CLI args, pass them through
# =============================================================================
# Inside main(), add to argparse:
#
#   parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
#   parser.add_argument("--no-cross-class-nms", action="store_true")
#
# And update the call to process_and_save_one_image_single_rank:
#
#   dt = process_and_save_one_image_single_rank(
#       ...                                   # all existing args unchanged
#       nms_iou_threshold=args.nms_iou_threshold,
#       cross_class_nms=not args.no_cross_class_nms,
#   )
