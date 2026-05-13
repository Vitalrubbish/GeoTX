# Training & Evaluation Guide

All **GeoTX** versions (v0.1, v0.2, v0.3) use the **image-conditioned SigmaSelector** (`selector_variant="v0.1"`), which produces per-(image, GPS) routing weights. A GPS-only variant exists as an ablation baseline.

## Model Definitions

| Name | `use_sigma_selector` | `selector_variant` | Routing Depends On |
|------|---------------------|-------------------|-------------------|
| **Baseline** | `False` | — | Uniform sum of 3 branches |
| **GPS-only** (ablation) | `True` | `None` | GPS coordinates only |
| **SigmaSelector** (all GeoTX) | `True` | `"v0.1"` | Image content + GPS |

## Three-Stage Training

### Stage 1: LoRA Training

Trains: LoRA adapters (ViT layers 18-23, q_proj/v_proj), image MLP, SigmaSelector, capsule heads.

```bash
# GeoTX (image-conditioned SigmaSelector)
python scripts/train_lora.py --mode full --epochs 10 \
    --lora-lr 1e-4 --location-lr 5e-5 \
    --queue-size 2048 --batch-size 32 \
    --selector-variant v0.1 \
    --output-dir outputs/lora

# Ablation: GPS-only SigmaSelector
python scripts/train_lora.py --mode full --epochs 10 \
    --lora-lr 1e-4 --location-lr 5e-5 \
    --queue-size 2048 --batch-size 32 \
    --output-dir outputs/lora
```

### Stage 2: SigmaSelector Fine-tuning

Trains: SigmaSelector only (all other params frozen).

```bash
# GeoTX (image-conditioned SigmaSelector)
python scripts/train_sigma_selector.py --mode full --epochs 10 \
    --lr 1e-3 --queue-size 2048 --batch-size 32 \
    --selector-variant v0.1 \
    --output-dir outputs/sigma_selector

# Ablation: GPS-only SigmaSelector
python scripts/train_sigma_selector.py --mode full --epochs 10 \
    --lr 1e-3 --queue-size 2048 --batch-size 32 \
    --output-dir outputs/sigma_selector
```

Optionally unfreeze capsule heads with `--unfreeze-capsule-head`.

### Stage 3: Negative Sampling

Trains: LoRA + MLP + SigmaSelector + capsule heads, with geographic negative masking.

```bash
# GeoTX (image-conditioned SigmaSelector)
python scripts/train_negative_sampling.py --mode full --epochs 10 \
    --lora-lr 1e-4 --location-lr 5e-5 \
    --queue-size 2048 --batch-size 32 \
    --neg-strategy threshold --neg-threshold 200 \
    --selector-variant v0.1 \
    --output-dir outputs/negative_sampling

# Ablation: GPS-only SigmaSelector
python scripts/train_negative_sampling.py --mode full --epochs 10 \
    --lora-lr 1e-4 --location-lr 5e-5 \
    --queue-size 2048 --batch-size 32 \
    --neg-strategy threshold --neg-threshold 200 \
    --output-dir outputs/negative_sampling
```

Also supports `--neg-strategy topk --neg-topk <N>`.

## Evaluation

All eval scripts auto-detect `selector_variant` from checkpoint metadata, so the same commands work for both GeoTX (v0.1) and GPS-only ablation.

### 1. Baseline (No SigmaSelector)

Full pretrained GeoCLIP — all three capsule branches summed equally.

```bash
python scripts/eval_sigma_selector.py \
    --dataset streetview_pano \
    --use-sigma-selector false
```

### 2. SigmaSelector Only (Stage 2 Checkpoint)

Evaluates only the SigmaSelector weights. The CLIP backbone, image MLP, and capsule bodies are frozen (as trained). Use the checkpoint from `train_sigma_selector.py`.

```bash
# GeoTX (image-conditioned, auto-detected from checkpoint)
python scripts/eval_sigma_selector.py \
    --dataset streetview_pano \
    --use-sigma-selector true \
    --selector-checkpoint outputs/sigma_selector/<run>/sigma_selector_best.pth

# Ablation: GPS-only
python scripts/eval_sigma_selector.py \
    --dataset streetview_pano \
    --use-sigma-selector true \
    --selector-checkpoint outputs/sigma_selector/<run>/sigma_selector_best.pth
```

Optionally add `--output-json data/streetview_pano/custom_name.json` to name the output.

### 3. SigmaSelector + LoRA (Stage 1 Checkpoint)

Evaluates LoRA adapters + image MLP + SigmaSelector + capsule heads together. Use the checkpoint from `train_lora.py`.

```bash
# GeoTX (image-conditioned, auto-detected from checkpoint)
python scripts/eval_lora.py \
    --dataset streetview_pano \
    --checkpoint outputs/lora/<run>/lora_best.pth

# Ablation: GPS-only
python scripts/eval_lora.py \
    --dataset streetview_pano \
    --checkpoint outputs/lora/<run>/lora_best.pth
```

### 4. Full Model (Stage 3 Checkpoint)

Evaluates the fully-trained model after negative sampling. All components (LoRA + MLP + SigmaSelector + capsule heads) trained with geographic negative masking. Use the checkpoint from `train_negative_sampling.py`.

```bash
# GeoTX (image-conditioned, auto-detected from checkpoint)
python scripts/eval_lora.py \
    --dataset streetview_pano \
    --checkpoint outputs/negative_sampling/<run>/neg_sampling_best.pth

# Ablation: GPS-only
python scripts/eval_lora.py \
    --dataset streetview_pano \
    --checkpoint outputs/negative_sampling/<run>/neg_sampling_best.pth
```

### Metrics

All eval scripts output accuracy at 5 distance thresholds plus mean error:

| Metric | Description |
|--------|-------------|
| `acc_1_km` | Fraction within 1 km |
| `acc_25_km` | Fraction within 25 km |
| `acc_200_km` | Fraction within 200 km |
| `acc_750_km` | Fraction within 750 km |
| `acc_2500_km` | Fraction within 2500 km |
| `mean_error_km` | Mean geodesic error distance |

Results are saved as JSON to `data/{dataset}/<name>_eval.json`.

### SigmaSelector Interpretability

```bash
python minipaper/figures/generate_demo3_plots.py
```

Generates three PDFs comparing urban vs. natural sigma weight distributions.

The image-conditioned SigmaSelector tests: **do urban images get different sigma routing than natural images?**

- If hypothesis is correct: urban → higher σ=256, natural → higher σ=1
- GPS-only ablation: weights nearly identical for urban vs. natural (no image content signal)

## Feasibility (Quick Prototyping)

```bash
# Generate subset (first time only)
python scripts/sample_streetview_subset.py

# GeoTX (image-conditioned SigmaSelector)
python scripts/train_lora.py --mode feasibility --epochs 10 --batch-size 16 \
    --selector-variant v0.1

# Ablation: GPS-only SigmaSelector
python scripts/train_lora.py --mode feasibility --epochs 10 --batch-size 16
```

## Checkpoint Compatibility

- v0.1 checkpoints include `selector_variant: "v0.1"` in saved dict
- `load_geotx_model()` auto-detects variant from checkpoint
- Old checkpoints (no `selector_variant` key) load as GPS-only
