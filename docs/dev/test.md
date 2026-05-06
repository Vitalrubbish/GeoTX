# Testing Document
## Quick Testing
```bash
python scripts/eval_lora.py --dataset streetview_pano --checkpoint geoclip/model/weights/neg_sampling_weights.pth
```

## All-Set Training And Testing
1. Initial Model: GeoCLIP
- Eval Script: 
    ```bash
    python scripts/eval_sigma_selector.py \
    --dataset streetview_pano \
    --use-sigma-selector false
    ```
2. GeoTX v0.1: SigmaSelector + unfrozen capsule heads.
- Train Script: 
    ```bash
    python scripts/train_sigma_selector.py \
    --mode full --epochs 10 --batch-size 32 \
    --unfreeze-capsule-head
    ```
- Eval Script: 
    ```bash
    python scripts/eval_sigma_selector.py \
    --dataset streetview_pano \
    --use-sigma-selector true \
    --selector-checkpoint outputs/sigma_selector/full_<timestamp>/selector_best.pth \
    --output-json data/streetview_pano/baseline_v1_eval.json
    ```
3. GeoTX v0.2: GeoTX v0.1 + LoRA on CLIP ViT last 6 layers + unfrozen image MLP
   - LoRA Configuration: r=4/8, alpha=8/16 (r×2), dropout=0.05
   - Target: q_proj, v_proj in ViT layers 18-23 (last 6 layers of ViT-L/14)
   - Trainable: LoRA adapters, image MLP, SigmaSelector, LocationEncoderCapsule heads
   - Frozen: CLIP ViT backbone (excluding LoRA), LocationEncoder backbone
   - Optimizer: Mixed precision (torch.cuda.amp), gradient checkpointing disabled
   - Learning Rates: 1e-4 for LoRA/MLP, 5e-5 for location modules
- Train Script:
    ```bash
    python scripts/train_lora.py \
    --mode full --epochs 10 --batch-size 32 \
    --lora-r 4 --lora-alpha 8 --lora-lr 1e-4 --location-lr 5e-5
    ```
- Eval Script:
    ```bash
    python scripts/eval_lora.py \
    --dataset streetview_pano \
    --checkpoint outputs/lora/full_<timestamp>/lora_best.pth
    ```
4. GeoTX v0.3: GeoTX v0.2 + Optimized Negative Sampling
- Train Script:
    ```bash
    python scripts/train_negative_sampling.py --mode full --epochs 20 --neg-strategy threshold --neg-threshold 200
    ```
- Eval Script:
    ```bash
    python scripts/eval_lora.py --dataset streetview_pano --checkpoint outputs/negative_sampling/full_threshold_<timestamp>/neg_sampling_best.pth
    ```
You can also use `topk` policy for negative sampling, however, we find out that threshold may have a better performance.


## Feasibility Check (both models)
1. Baseline
    ```bash
    python scripts/train_sigma_selector.py \
    --mode feasibility --epochs 10 --batch-size 16 \
    --unfreeze-capsule-head
    ```
2. New Model (900 train / 100 val subset)
    ```bash
    python scripts/train_lora.py \
    --mode feasibility --epochs 10 --batch-size 16 \
    --lora-r 4 --lora-alpha 8 --lora-lr 1e-4 --location-lr 5e-5
    ```

**Notice**: You can modify any parameters as you like.