import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Explicitly set it if needed
if 'HF_HUB_CACHE' in os.environ:
    os.environ['HF_HOME'] = os.environ['HF_HUB_CACHE']  # Some libraries use HF_HOME instead
    print(f"Set HF_HOME to {os.environ['HF_HOME']}")
    print(f"HF_HUB_CACHE is set to {os.environ['HF_HUB_CACHE']}")

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sam3.model.sam3_image_processor import Sam3Processor

from .utils.image import ImageCropper, load_images_from_dir
from .utils.dataloader import create_distributed_dataloader
from .utils.model import extract_masks_scores
from .utils.processor import set_tensor_batch
from .utils.performance import TimingContext, PerformanceTracker

logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        with TimingContext("Initializing distributed process group", rank):
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            torch.cuda.set_device(local_rank)
            
            # Verify communication
            if rank == 0:
                logger.info(f"Master address: {os.environ.get('MASTER_ADDR', 'localhost')}:{os.environ.get('MASTER_PORT', '29500')}")
            
            # Test communication
            test_tensor = torch.tensor([rank], device=f'cuda:{local_rank}')
            dist.all_reduce(test_tensor)
            
            if rank == 0:
                logger.info("✓ Cross-node communication verified")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_results_from_all_ranks(
    local_results: Dict[str, Dict[str, torch.Tensor]],
    world_size: int,
    rank: int,
    device: torch.device
) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    """Gather results from all ranks to rank 0."""
    if world_size == 1:
        return local_results
    
    gathered_results = {}
    
    for prompt in local_results.keys():
        gathered_results[prompt] = {"masks": [], "scores": [], "indices": []}
        
        for key in ["masks", "scores", "indices"]:
            local_tensor = local_results[prompt][key]

            # Convert indices list to tensor if needed
            if key == "indices":
                local_tensor = torch.tensor(local_tensor, device=device)
            
            # Gather tensor sizes from all ranks
            local_size = torch.tensor([local_tensor.shape[0]], device=device)
            size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            
            # Prepare to gather tensors
            if rank == 0:
                gathered_tensors = []
                total_size = sum(s.item() for s in size_list)
                data_size_mb = (total_size * np.prod(local_tensor.shape[1:]) * 4) / (1024**2)
                
                logger.info(f"  Gathering '{prompt}' {key}: {data_size_mb:.1f} MB")
                
                for size in size_list:
                    gathered_tensors.append(
                        torch.zeros(
                            (size.item(), *local_tensor.shape[1:]),
                            dtype=local_tensor.dtype,
                            device=device
                        )
                    )
            else:
                gathered_tensors = None
            
            # Gather tensors to rank 0
            gather_start = time.time()
            if rank == 0:
                gathered_tensors[0].copy_(local_tensor)
                for src_rank in range(1, world_size):
                    dist.recv(gathered_tensors[src_rank], src=src_rank)
            else:
                dist.send(local_tensor, dst=0)
            
            if rank == 0:
                gather_time = time.time() - gather_start
                if world_size > 1 and gather_time > 0.1:
                    bandwidth = data_size_mb / gather_time
                    logger.info(f"    Transfer time: {gather_time:.2f}s | {bandwidth:.0f} MB/s")
                
                gathered_results[prompt][key] = torch.cat(gathered_tensors, dim=0)
    
    return gathered_results if rank == 0 else None


def process_multiple_prompts_distributed(
    images_tensor: torch.Tensor,
    processor,
    prompts: List[str],
    output_dir: str,
    class_colors: Dict[str, int],
    patch_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device = None,
    tracker: PerformanceTracker = None,
) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    """Process images with multiple text prompts in a distributed manner."""
    N, C, H, W = images_tensor.shape
    
    if rank == 0:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Create patches
    with TimingContext("Creating patches", rank) as timer:
        image_cropper = ImageCropper(
            qlty_window=patch_size,
            qlty_step=patch_size,
            qlty_border=0,
            qlty_border_weight=0.0,
        )
        patches = image_cropper.unstitch(images_tensor.cpu())
        
        if rank == 0:
            print(f"  Patch shape: {patches.shape}", flush=True)
            logger.info(f"  Total patches: {patches.shape[0]} ({patches.shape[0]//N} per image)")
            if world_size > 1:
                patches_per_gpu = patches.shape[0] // world_size
                logger.info(f"  Patches per GPU: ~{patches_per_gpu} ({patches_per_gpu/patches.shape[0]*100:.1f}% of total)")
    
    if tracker:
        tracker.record('patch_creation', timer.elapsed())

    # Create distributed dataloader
    dataloader = create_distributed_dataloader(
        patches=patches,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        world_size=world_size,
        rank=rank
    )

    print(f"[RANK {rank}] Dataloader has {len(dataloader)} batches", flush=True)

    # Process all batches
    local_results = {p: {"masks": [], "scores": [], "indices": []} for p in prompts}
    
    inference_start = time.time()
    for batch_idx, batch_patches in enumerate(dataloader):
        batch_start = time.time()
        
        if rank == 0 and batch_idx % 10 == 0:
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        
        batch_patches = batch_patches.to(device)
        actual_batch_size = batch_patches.shape[0]  # ADDED: Handle variable batch size
        
        for prompt_idx, prompt in enumerate(prompts):
            prompt_start = time.time()
            
            state = processor.set_tensor_batch(batch_patches)
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(state=state, prompt=prompt)
            
            batch_masks, batch_scores = extract_masks_scores(output)
            
            # MODIFIED: Process each patch in the batch separately
            for idx_in_batch in range(actual_batch_size):
                # FIXED: Calculate correct global index for THIS patch
                global_idx = rank + (batch_idx * batch_size + idx_in_batch) * world_size
                
                # ADDED: Extract masks for this specific patch
                if batch_masks is not None and batch_masks.shape[0] > 0:
                    # Handle different output shapes
                    if batch_masks.ndim == 5:  # [num_detections, batch_size, 1, H, W]
                        patch_masks = batch_masks[:, idx_in_batch, :, :, :]
                        patch_scores = batch_scores[:, idx_in_batch] if batch_scores.ndim > 1 else batch_scores
                    elif batch_masks.ndim == 4:  # [batch_size, 1, H, W]
                        patch_masks = batch_masks[idx_in_batch:idx_in_batch+1]
                        patch_scores = batch_scores[idx_in_batch:idx_in_batch+1] if batch_scores.ndim > 0 else batch_scores
                    else:
                        patch_masks = batch_masks[0:1] if batch_masks.shape[0] > idx_in_batch else batch_masks
                        patch_scores = batch_scores[0:1] if batch_scores.shape[0] > idx_in_batch else batch_scores
                    
                    # Aggregate multiple detections for this patch
                    if patch_masks.shape[0] > 1:
                        combined_mask = patch_masks.max(dim=0, keepdim=True)[0]
                        if patch_scores.ndim > 0 and patch_scores.shape[0] > 1:
                            combined_score = patch_scores.max(dim=0, keepdim=True)[0]
                        else:
                            combined_score = patch_scores.max() if patch_scores.numel() > 0 else torch.tensor([0.0], device=device)
                        if combined_score.ndim == 0:
                            combined_score = combined_score.unsqueeze(0)
                    else:
                        combined_mask = patch_masks[0:1] if patch_masks.ndim > 3 else patch_masks.unsqueeze(0)
                        combined_score = patch_scores[0:1] if patch_scores.ndim > 0 else patch_scores.unsqueeze(0)
                        
                        # Ensure correct dimensions
                        if combined_mask.ndim == 3:  # [1, H, W]
                            combined_mask = combined_mask.unsqueeze(1)  # [1, 1, H, W]
                else:
                    # No detections - create empty placeholder
                    combined_mask = torch.zeros((1, 1, patch_size, patch_size), device=device, dtype=torch.float32)
                    combined_score = torch.zeros((1,), device=device, dtype=torch.float32)
                
                local_results[prompt]["masks"].append(combined_mask)
                local_results[prompt]["scores"].append(combined_score)
                local_results[prompt]["indices"].append(global_idx)
            
            if rank == 0 and batch_idx % 10 == 0:
                prompt_time = time.time() - prompt_start
                logger.info(f"  Prompt '{prompt}' [{prompt_time:.2f}s]")
        
        if rank == 0 and batch_idx % 10 == 0:
            batch_time = time.time() - batch_start
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)} completed [{batch_time:.2f}s]")
    
    inference_time = time.time() - inference_start
    
    if rank == 0:
        logger.info(f"All batches processed [{TimingContext._format_time(inference_time)}]")
    
    if tracker:
        tracker.record('inference', inference_time)
    
    print(f"[RANK {rank}] Finished inference. Concatenating local results...", flush=True)

    # Concatenate local results
    for prompt in prompts:
        local_results[prompt]["masks"] = torch.cat(local_results[prompt]["masks"], dim=0)
        local_results[prompt]["scores"] = torch.cat(local_results[prompt]["scores"], dim=0)
        
        print(f"[RANK {rank}] Prompt '{prompt}': {local_results[prompt]['masks'].shape[0]} patches", flush=True)
    
    # Gather results
    with TimingContext("Gathering results from all GPUs", rank) as timer:
        gathered_results = gather_results_from_all_ranks(
            local_results, world_size, rank, device
        )
    
    if tracker:
        tracker.record('gathering', timer.elapsed())
    
    # Only rank 0 stitches and saves
    if rank == 0:
        # Reorder results by global index
        for prompt in prompts:
            indices_tensor = gathered_results[prompt]["indices"]
            sort_order = torch.argsort(indices_tensor)
            
            gathered_results[prompt]["masks"] = gathered_results[prompt]["masks"][sort_order]
            gathered_results[prompt]["scores"] = gathered_results[prompt]["scores"][sort_order]
            
            print(f"[RANK {rank}] After sorting - Prompt '{prompt}': {gathered_results[prompt]['masks'].shape}", flush=True)
        
        with TimingContext("Stitching results", rank) as timer:
            stitched_results = {}
            patches_per_image = (H // patch_size) * (W // patch_size)
            expected_total = N * patches_per_image
            
            for prompt in prompts:
                masks = gathered_results[prompt]["masks"]
                scores = gathered_results[prompt]["scores"]

                print(f"[RANK {rank}] Prompt '{prompt}':", flush=True)
                print(f"  Got {masks.shape[0]} masks, expected {expected_total}", flush=True)
                
                # ADDED: Verify we have the right number
                if masks.shape[0] != expected_total:
                    logger.warning(f"  WARNING: Expected {expected_total} masks but got {masks.shape[0]}")
                    if masks.shape[0] > expected_total:
                        masks = masks[:expected_total]
                        scores = scores[:expected_total]
                    elif masks.shape[0] < expected_total:
                        padding_needed = expected_total - masks.shape[0]
                        masks = torch.cat([
                            masks,
                            torch.zeros((padding_needed, *masks.shape[1:]), device=masks.device, dtype=masks.dtype)
                        ], dim=0)
                        scores = torch.cat([
                            scores,
                            torch.zeros((padding_needed, *scores.shape[1:]), device=scores.device, dtype=scores.dtype)
                        ], dim=0)
                
                # Prepare for stitching
                masks_for_stitch = masks.cpu().float().numpy().astype(np.float32)
                
                # MODIFIED: Better score expansion
                if scores.ndim == 1:
                    scores_for_stitch = scores.view(-1, 1, 1, 1).expand(-1, 1, patch_size, patch_size)
                else:
                    scores_for_stitch = scores
                
                scores_for_stitch = scores_for_stitch.cpu().float().numpy().astype(np.float32)
                
                print(f"  masks_for_stitch shape: {masks_for_stitch.shape}", flush=True)
                print(f"  scores_for_stitch shape: {scores_for_stitch.shape}", flush=True)
                
                stitched_results[prompt] = {
                    "stitched_mask": image_cropper.stitch(masks_for_stitch),
                    "stitched_score": image_cropper.stitch(scores_for_stitch)
                }
        
        if tracker:
            tracker.record('stitching', timer.elapsed())
        
        # Save combined masks
        with TimingContext("Creating combined segmentation maps", rank) as timer:
            combined_dir = output_path / "combined"
            combined_dir.mkdir(exist_ok=True)
            
            for i in range(N):
                combined_mask = np.zeros((H, W), dtype=np.uint8)
                
                for prompt, class_id in class_colors.items():
                    mask = stitched_results[prompt]["stitched_mask"][i]
                    
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()
                    
                    mask = mask.squeeze()
                    
                    binary_mask = mask > 0.5
                    combined_mask[binary_mask] = class_id
                
                Image.fromarray(combined_mask).save(
                    combined_dir / f"combined_{i:04d}.tif",
                    compression="tiff_deflate"
                )
            
            logger.info(f"✓ Saved {N} combined masks to {combined_dir}")
        
        if tracker:
            tracker.record('save_combined', timer.elapsed())
        
        # Save individual masks
        with TimingContext("Saving individual masks", rank) as timer:
            for prompt in prompts:
                prompt_dir = output_path / prompt.replace(" ", "_")
                prompt_dir.mkdir(exist_ok=True)
                
                for i in range(N):
                    mask = stitched_results[prompt]["stitched_mask"][i]
                    
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()
                    
                    mask = mask.squeeze()
                    
                    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
                    
                    Image.fromarray(mask_uint8).save(
                        prompt_dir / f"mask_{i:04d}.tif",
                        compression="tiff_deflate"
                    )
            
            logger.info(f"✓ All individual masks saved")
        
        if tracker:
            tracker.record('save_individual', timer.elapsed())
        
        return stitched_results
    
    return None


def main():
    parser = argparse.ArgumentParser(description="SAM3 Distributed Segmentation Inference")
    
    # Paths
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input TIFF images")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save segmentation results")
    
    # Inference parameters
    parser.add_argument("--patch-size", type=int, default=512, help="Size of patches for processing")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for predictions")
    
    # Prompts
    parser.add_argument(
        "--prompts",
        nargs="+", 
        default=["cortex", "Phloem Fibers", "Xylem vessels", "Pith cells", "outer cells"],
        help="Text prompts to use for segmentation"
    )
    
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Setup logging
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    logger.info(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    from sam3.model_builder import build_sam3_image_model

    # Initialize performance tracker
    tracker = PerformanceTracker(rank)

    # Generate class colors
    CLASS_COLORS = {prompt: idx + 1 for idx, prompt in enumerate(args.prompts)}
    
    # Validate inputs
    if rank == 0:
        logger.info("Class ID mapping:")
        for prompt, class_id in CLASS_COLORS.items():
            logger.info(f"  {class_id}: '{prompt}'")
        
        if not os.path.isdir(args.input_dir):
            logger.error(f"Error: Input directory not found at {args.input_dir}")
            sys.exit(1)
    
    # Synchronize
    if world_size > 1:
        dist.barrier()
    
    # Load images
    with TimingContext("Loading images", rank) as timer:
        images_tensor = load_images_from_dir(args.input_dir)
        if rank == 0:
            logger.info(f"  Loaded {images_tensor.shape[0]} images")
            logger.info(f"  Image shape: {images_tensor.shape}")
    
    if tracker:
        tracker.record('image_loading', timer.elapsed())
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        logger.info(f"Using {world_size} GPU(s)")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
    
    try:
        # Load model
        if rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("Loading SAM3 model...")
            logger.info("=" * 60)
        
        model_start = time.time()
        model = build_sam3_image_model()
        model = model.to(device)
        
        # Wrap with DDP
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            if rank == 0:
                logger.info("✓ Model wrapped with DDP")
        
        model.eval()
        
        # Setup processor
        Sam3Processor.set_tensor_batch = set_tensor_batch
        actual_model = model.module if isinstance(model, DDP) else model
        processor = Sam3Processor(actual_model, confidence_threshold=args.confidence)
        
        model_time = time.time() - model_start
        if rank == 0:
            logger.info(f"✓ Model loaded successfully [{TimingContext._format_time(model_time)}]")
        
        if tracker:
            tracker.record('model_loading', model_time)
        
        # Process
        if rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("Starting distributed segmentation...")
            logger.info("=" * 60)
        
        num_patches = (images_tensor.shape[2] // args.patch_size) * (images_tensor.shape[3] // args.patch_size) * images_tensor.shape[0]
        
        results = process_multiple_prompts_distributed(
            images_tensor=images_tensor,
            processor=processor,
            prompts=args.prompts,
            output_dir=args.output_dir,
            class_colors=CLASS_COLORS,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            rank=rank,
            world_size=world_size,
            device=device,
            tracker=tracker,
        )
        
        if rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("✓ Segmentation completed successfully!")
            logger.info(f"Results saved to: {args.output_dir}")
            logger.info("=" * 60)
            
            # Log performance summary
            tracker.log_summary(
                world_size=world_size,
                num_images=images_tensor.shape[0],
                num_patches=num_patches,
                num_prompts=len(args.prompts)
            )
        
    except Exception as e:
        if rank == 0:
            logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
