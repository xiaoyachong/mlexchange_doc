import lightly_train
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os

# ------------------------
# Class definitions
# ------------------------
CLASS_COLORS = {
    0: [128,128,128],    # gray - background
    1: [0, 0, 255],      # Blue - Cortex
    2: [0, 255, 0],      # Green - Phloem Fibers
    3: [128, 0, 128],    # Purple - Phloem
    4: [255, 0, 0],      # Red - Xylem vessels
    5: [255, 255, 0],    # Yellow - Air-based Pith cells
    6: [255, 165, 0],    # Orange - Water-based Pith cells
}

CLASS_NAMES = {
    0: "background",
    1: "Cortex",
    2: "Phloem Fibers",
    3: "Phloem",
    4: "Xylem vessels",
    5: "Air-based Pith cells",
    6: "Water-based Pith cells",
}

IGNORE_CLASSES = [255]

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def mask_to_color(mask_array):
    """Convert a grayscale mask to RGB using class colors."""
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask_array == class_id] = color

    return color_mask

def normalize_image_percentile(img_array, lower_percentile=2, upper_percentile=98):
    """Normalize image using percentiles for high contrast display."""
    lower = np.percentile(img_array, lower_percentile)
    upper = np.percentile(img_array, upper_percentile)

    img_clipped = np.clip(img_array, lower, upper)

    if upper > lower:
        img_normalized = ((img_clipped - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        img_normalized = img_array.astype(np.uint8)

    return img_normalized

# ------------------------
# Dataset class
# ------------------------
class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = Path(img_dir)

        IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        self.img_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        with Image.open(img_path) as img:
            if img.mode not in ['RGB', 'L', 'RGBA']:
                if img.mode in ['I', 'I;16', 'F']:
                    img_array = np.array(img)
                    img_array = ((img_array - img_array.min()) /
                                (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                else:
                    img = img.convert('L')

            img = img.convert("RGB")
            img_np = np.array(img)

        img_tensor = transforms.ToTensor()(img)  # [0, 1] range

        return {
            'image': img_tensor,
            'image_np': img_np,
            'img_path': str(img_path),
            'img_name': img_path.name
        }

# ------------------------
# Batch prediction function (unchanged — still used for local nn.Module path)
# ------------------------
def predict_batch_correct(model, batch_images):
    """
    Run batch inference matching model.predict() behaviour exactly.

    Args:
        model: The loaded raw nn.Module
        batch_images: Tensor of shape (B, C, H, W) - raw images [0-1] range

    Returns:
        List of predicted masks (one per image) as numpy arrays
    """
    if model.training:
        model.eval()

    device = next(model.parameters()).device
    batch_size = batch_images.shape[0]
    pred_masks = []

    for i in range(batch_size):
        x = batch_images[i]  # (C, H, W)
        image_h, image_w = x.shape[-2:]

        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        x = transforms.functional.normalize(
            x, mean=model.image_normalize["mean"], std=model.image_normalize["std"]
        )

        crop_size = min(model.image_size)
        x = transforms.functional.resize(x, size=[crop_size])
        x = x.unsqueeze(0)  # (1, C, H', W')

        logits = model._forward_logits(x)  # (1, K+1, H', W')
        logits = logits[:, :-1]            # (1, K, H', W')

        logits = F.interpolate(
            logits, size=(image_h, image_w), mode="bilinear"
        )  # (1, K, H, W)

        masks = logits.argmax(dim=1)                        # (1, H, W)
        masks = model.internal_class_to_class[masks]        # (1, H, W)

        pred_masks.append(masks[0].cpu().numpy())

    return pred_masks


# =============================================================================
# NEW: model loading helper (MLflow or local)
# =============================================================================
def load_dino(
    ckpt_path: str,
    device: torch.device,
    mlflow_model_name: str = None,
    mlflow_model_version: str = None,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Load DINO — from MLflow registry if mlflow_model_name is given,
    otherwise fall back to the original lightly_train local checkpoint.
    Returns either an MLflow pyfunc model or a raw nn.Module.
    Both are handled transparently by _call_dino().
    """
    if mlflow_model_name:
        from mlex_utils.mlflow_utils.mlflow_model_client import MLflowModelClient
        client = MLflowModelClient()
        model = client.load_model(mlflow_model_name, version=mlflow_model_version)
        if rank == 0:
            print(
                f"Loaded DINO from MLflow: {mlflow_model_name} "
                f"v{mlflow_model_version or 'latest'}"
            )
        return model  # mlflow pyfunc model

    # Original local path (unchanged behaviour)
    model = lightly_train.load_model(ckpt_path)
    model.eval()
    model = model.to(device)
    if rank == 0:
        print(f"Model loaded from local checkpoint on {world_size} GPU(s)")
    return model  # raw nn.Module


# =============================================================================
# NEW: unified inference call (handles both pyfunc and nn.Module)
# =============================================================================
def _call_dino(model, batch_images: torch.Tensor) -> list:
    """
    Unified call for MLflow pyfunc model and raw nn.Module.
    Returns list of numpy uint8 (H, W) arrays — same as predict_batch_correct().

    Args:
        model       : pyfunc model (MLflow) OR raw nn.Module
        batch_images: torch.Tensor (B, C, H, W) float32 [0, 1] on any device
    """
    import mlflow.pyfunc
    if isinstance(model, mlflow.pyfunc.PyFuncModel):
        images_np = batch_images.cpu().numpy()      # (N, C, H, W) float32
        result = model.predict({"images": images_np})
        masks_np = result["masks"]                  # (N, H, W) uint8
        return [masks_np[i] for i in range(masks_np.shape[0])]
    else:
        return predict_batch_correct(model, batch_images)


# ------------------------
# Main function
# ------------------------
def main():
    parser = argparse.ArgumentParser(description='Distributed DINO inference')
    parser.add_argument('--input-dir',  type=str, required=True,  help='Input image directory')
    parser.add_argument('--output-dir', type=str, required=True,  help='Output directory')
    parser.add_argument('--finetuned-checkpoint', type=str, required=False, default=None,
                        help='Path to finetuned checkpoint (not required when using --mlflow-model-name)')
    parser.add_argument('--save-overlay', action='store_true', help='Save overlay images')
    parser.add_argument('--batch-size',   type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num-workers',  type=int, default=4, help='Number of dataloader workers')

    # ── NEW: MLflow model loading ────────────────────────────────────────────
    parser.add_argument('--mlflow-model-name',    type=str, default=None,
                        help='MLflow registered model name for DINO. '
                             'When set, --finetuned-checkpoint is ignored.')
    parser.add_argument('--mlflow-model-version', type=str, default=None,
                        help='MLflow model version (default: latest).')
    # ── END NEW ──────────────────────────────────────────────────────────────

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Setup paths
    IMG_DIR    = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_MASK_DIR    = OUTPUT_DIR / "semantic_masks"
    OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "semantic_masks_overlay"
    CKPT_PATH  = args.finetuned_checkpoint

    # Create output directories (only on rank 0)
    if rank == 0:
        OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
        if args.save_overlay:
            OUTPUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # ── CHANGED: load_dino() replaces the original 3-line load block ─────────
    if rank == 0:
        print("Loading model...")
    model = load_dino(
        ckpt_path=CKPT_PATH,
        device=device,
        mlflow_model_name=args.mlflow_model_name,       # ← NEW
        mlflow_model_version=args.mlflow_model_version, # ← NEW
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        print(f"Model loaded on {world_size} GPU(s)")
    # ── END CHANGE ────────────────────────────────────────────────────────────

    # Setup dataset and dataloader with distributed sampler
    dataset = ImageDataset(IMG_DIR)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler else False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if rank == 0:
        print(f"Total images: {len(dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Number of GPUs: {world_size}")
        print(f"Images per GPU: {len(dataset) // world_size}")
        print("-" * 70)

    # Batch inference + save
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            batch_images = batch['image'].to(device)

            # ── CHANGED: was predict_batch_correct(model, batch_images) ──────
            pred_masks = _call_dino(model, batch_images)
            # ── END CHANGE ────────────────────────────────────────────────────

            for i in range(len(batch['img_name'])):
                img_np = batch['image_np'][i]
                if isinstance(img_np, torch.Tensor):
                    img_np = img_np.cpu().numpy()

                pred_mask_np = pred_masks[i]  # Already numpy
                img_name = batch['img_name'][i]

                pred_unique = np.unique(pred_mask_np)

                if rank == 0:
                    print(f"GPU {rank}: {img_name}")
                    print(f"  Predicted classes: {pred_unique}")

                # Save mask as 1-channel grayscale (class indices)
                mask_save_path = OUTPUT_MASK_DIR / img_name
                Image.fromarray(pred_mask_np.astype(np.uint8), mode='L').save(mask_save_path)

                if rank == 0:
                    print(f"  Saved mask to: {mask_save_path}")

                if args.save_overlay:
                    pred_color_mask = mask_to_color(pred_mask_np)
                    img_np_normalized = normalize_image_percentile(
                        img_np, lower_percentile=2, upper_percentile=98
                    )
                    pred_overlay = (0.6 * img_np_normalized + 0.4 * pred_color_mask).astype(np.uint8)
                    overlay_save_path = OUTPUT_OVERLAY_DIR / img_name
                    Image.fromarray(pred_overlay).save(overlay_save_path)

                    if rank == 0:
                        print(f"  Saved overlay to: {overlay_save_path}")

                if rank == 0:
                    print("-" * 70)

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print(f"\nInference complete!")
        print(f"Masks saved to: {OUTPUT_MASK_DIR}")
        if args.save_overlay:
            print(f"Overlays saved to: {OUTPUT_OVERLAY_DIR}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
