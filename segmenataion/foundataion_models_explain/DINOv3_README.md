# DINOv3

DINOv3 is a self-supervised vision foundation model pretrained on ~1.7B curated web images, producing general-purpose dense features without task-specific fine-tuning. These frozen features feed into a lightweight, task-specific head — in our case, an EoMT segmentation head for semantic segmentation.

## Training Data Summary

| Component | Size | Description |
|---|---|---|
| Raw data pool | ~17B images | Public Instagram posts, platform-level content-moderated |
| LVD-1689M (Part 1) | 1,689M images | Curated from the raw pool via hierarchical k-means clustering (5 levels, 200M→25k clusters) on DINOv2 embeddings, then balanced sampling for even concept coverage |
| Retrieval-curated (Part 2) | unspecified subset | Images retrieved from the raw pool by similarity to seed datasets relevant to downstream tasks |
| Benchmark datasets (Part 3) | standard sizes | ImageNet-1k, ImageNet-22k, and Mapillary Street-Level Sequences (street-view images) |
| SAT-493M (separate satellite model, 7B, not used for our backbones) | 493M images | 512×512 tiles from Maxar RGB ortho-rectified satellite imagery at 0.6m resolution |

The web model (top four rows, combined) is what our fine-tuned backbones are initialized from — checkpoint filenames use `lvd1689m`, confirming this. SAT-493M is a separate, domain-specific model not relevant to our work.

Source: DINOv3 technical report (Siméoni et al., 2025), arXiv:2508.10104.
