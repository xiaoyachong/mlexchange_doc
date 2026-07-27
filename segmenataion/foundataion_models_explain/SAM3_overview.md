# SAM 3 Overview

SAM 3 is a foundation model for promptable, concept-level segmentation of images and videos — given a short text phrase or image exemplar, it detects, segments, and tracks every matching instance — trained on the SA-Co dataset comprising over 50 million images and videos with more than 4 million unique concept labels and roughly 1.4 billion masks.

## Training Data

| Dataset | Type | # Images/Videos | # Unique NPs | # Image/Video-NP pairs | # Masks/Masklets | Description |
|---|---|---|---|---|---|---|
| SA-Co/HQ | Image | 5.2M | 4.0M | 146.1M | 52.3M masks | Web, stock, egocentric, robotics, art photos |
| SA-Co/SYN | Image | 39.4M | 38.0M | 1.7B | 1.4B masks | Web-scraped MetaCLIP images with captions |
| SA-Co/EXT | Image | 9.3M | 497.4K | 136.6M | 70.5M masks | Driving, medical, wildlife, fashion imagery |
| SA-Co/VIDEO | Video | 52.5K | 24.8K | 134.3K | 467.1K masklets | Everyday, egocentric, wildlife video clips |
| SA-1B | Image (PVS/aux) | ~11M | – | – | ~1.1B masks | Diverse stock photo images |
| SA-V / LVOSv2 | Video (PVS/aux) | – | – | – | – | General object-tracking video clips |

*NP = noun phrase, the short text label used to prompt SAM 3 (e.g. "yellow school bus," "striped cat"). SAM 3 restricts concept prompts to simple noun phrases rather than full sentences or referring expressions.*

*Source: SAM 3 paper, Table 24, Table 25, and §5/§E.1.*
