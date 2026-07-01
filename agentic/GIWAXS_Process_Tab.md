# GIWAXS Process Tab — Step-by-Step Explanation

## 1. Q-Map Resolution (always runs)

Sets the output Q-map size in pixels (default 512×512, up to 2000×2000). This is the
core step — converts the raw detector image into Q-space using pyFAI.

Under the collapsible **Geometry Parameters** you configure the physical setup:

| Parameter | Description |
|-----------|-------------|
| SDD (mm) | Sample-to-detector distance |
| Energy (keV) | X-ray beam energy |
| Beam X / Beam Y (px) | Beam center position on the detector |
| Incident Angle (°) | Grazing incidence angle |
| Geometry | GIWAXS or GISAXS |

---

## 2. Feature Extraction (collapsible, optional)

- **Compute intensity statistics** — mean, min, max, std of the image intensity
- **Compute anisotropy** — measures how directional/oriented the scattering pattern is (Herman's orientation factor)
- **Detect peaks (count)** — counts the number of scattering peaks found

---

## 3. ML / VAE Encoding (open by default)

- **Precompute all UMAP embeddings** — pre-generates 2D projections for every view
  the Explore tab can show (features, pixels, Q-map, radial, azimuthal, polar,
  sectors, multi-scale). Adds processing time upfront but makes browsing instant.
  Configurable via `n_neighbors` and `min_dist`.
- **Precompute search profiles** — stores a fixed-grid I(|q|) radial profile,
  downsampled Q-map, and peak catalog per image so Ring Search queries are fast.
- **Run MLExchange VAE encoding** — encodes images using a variational autoencoder
  (input can be raw, log-scale raw, or Q-map). Optionally also computes UMAP
  coordinates from the VAE latent vectors.

---

## 4. Vision Model — Batch (open by default, optional)

Runs every image through an Ollama/vLLM vision model after Q-map processing.

| Option | Description |
|--------|-------------|
| Vision model | Image → text description (Ollama or vLLM via SSH tunnel) |
| Embedding model | Text → vector stored as `text_embedding` in Tiled metadata |
| Prompt | Customizable instruction sent to the vision model |
| Save mode | Append to or replace previous vision entries |
| Timeout | Per-image timeout (2 min to 30 min) |

Requires Ollama running on the host and the API server (port 8002).

---

## 5. GIWAXS Vision Pipeline — Structured (open by default, optional)

A more structured multi-step pipeline per image:

1. Caking (azimuthal rebinning)
2. 1D integration + peak fitting
3. χ-cuts + Herman's orientation
4. Spot detection
5. (Optional) VLM text synthesis — a natural-language summary per image

Requires a PONI calibration file — set `poni_path` in frame metadata or
`GIWAXS_VISION_DEFAULT_PONI` in `.env`.

Results are richer and more structured than the batch vision step above.

---

## Processing Flow Summary

```
Select samples (Select tab)
        ↓
Q-map conversion  ←── geometry params, resolution
        ↓
Feature extraction  ←── intensity stats, anisotropy, peaks
        ↓
ML / VAE  ←── UMAP embeddings, search profiles, VAE latent vectors
        ↓
Vision (batch)  ←── free-form image description via vision LLM
        ↓
Vision pipeline  ←── structured: cake → 1D → peaks → orientation → synthesis
        ↓
Open in Viewer
```
