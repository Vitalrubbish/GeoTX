# SigmaSelector: Image-Conditioned Routing

## Design

The SigmaSelector replaces GeoCLIP's equal-sum fusion of three RFF branches with a **learnable, image-conditioned weighted sum**. For each (image, GPS) pair, it predicts routing weights that determine how much each spatial-frequency branch contributes.

```
[image_features (512) | projected_gps (2)]  →  Linear(514→128) → ReLU → Linear(128→64) → ReLU → Linear(64→3) → Softmax
```

This implements the research hypothesis:

> Urban/detailed scenes should route toward fine-grained (high σ) encoding;
> natural/landscape scenes should route toward coarse (low σ) encoding.

## Per-Pair Routing

Each (image_i, gps_j) pair gets its own routing weights. A skyscraper photo and a forest photo at the **same GPS coordinate** get **different** sigma routing — the model adapts its location encoding to the visual content.

| Model | Input | Output Shape | Routing Depends On |
|-------|-------|-------------|-------------------|
| Baseline (no selector) | N/A | — | Uniform sum of 3 branches |
| **SigmaSelector** | gps_proj (M, 2) + img_feat (N, 512) | (N, M, 3) | Image content + GPS |

## Computational Cost

The expensive capsule branches (RFF + 3-layer MLPs) are unchanged. The SigmaSelector MLP (~74K params) processes N×M pairs — negligible overhead (~0.5ms for N=32, M=2080 on GPU). For large galleries (e.g. 100K), the forward pass chunks over GPS locations to keep memory bounded.

## API

```python
# Enable SigmaSelector (image-conditioned)
model = GeoCLIP(use_sigma_selector=True, selector_variant="v0.1")

# Interpretability: extract per-pair routing weights
weights = model.location_encoder.get_sigma_weights(gps, image_features=img_feat)
# weights.shape = (N, M, 3) — one set of 3 weights per (image, GPS) pair
```

## GPS-only Ablation

For comparison, the codebase also retains a **GPS-only** variant (`SigmaSelector` class) that takes only projected GPS coordinates and outputs location-only weights of shape (M, 3). This variant cannot distinguish between two different images at the same location.

```python
# GPS-only ablation (selector_variant=None, the default)
model = GeoCLIP(use_sigma_selector=True)  # selector_variant defaults to None
weights = model.location_encoder.get_sigma_weights(gps)
# weights.shape = (M, 3) — same weights for any image at that GPS
```

| Variant | Class | Input | Output | Use Case |
|---------|-------|-------|--------|----------|
| **SigmaSelector** (GeoTX) | `ImageConditionedSigmaSelector` | img (512) + gps (2) | (N, M, 3) | All GeoTX v0.1–v0.3 |
| GPS-only ablation | `SigmaSelector` | gps (2) | (M, 3) | Baseline comparison |

## Implementation

| File | Change |
|------|--------|
| `geoclip/model/location_encoder.py` | `ImageConditionedSigmaSelector` class (line 66); `SigmaSelector` (GPS-only, line 46); `LocationEncoder` accepts `selector_variant` and optional `image_features` |
| `geoclip/model/GeoCLIP.py` | `__init__` accepts `selector_variant`; `forward()` pipes image features, handles 3D output via chunked `_forward_v01()` |
| `scripts/train_lora.py` | `--selector-variant` CLI arg |
| `scripts/train_sigma_selector.py` | `--selector-variant` CLI arg |
| `scripts/train_negative_sampling.py` | `--selector-variant` CLI arg |
| `demos/demo_utils.py` | `load_geotx_model()` auto-detects variant from checkpoint |
| `demos/demo3_*.ipynb` | Updated weight extraction for image-conditioned API |
| `minipaper/figures/generate_demo3_plots.py` | Updated weight extraction and titles |

## Backward Compatibility

- `selector_variant=None` (default): GPS-only behavior, all existing code unchanged
- `selector_variant="v0.1"`: image-conditioned routing used by all GeoTX versions
- Existing pretrained weights load with `strict=False` (unchanged)
- Old checkpoints without `selector_variant` key load as GPS-only
