# GeoCLIP Enhancement via LoRA on ImageEncoder

## 1. Problem Statement
Previous improvements focused on the **location_side** by introducing the *Coordinate-conditioned Attention* (SigmaSelector) and unfreezing the terminal layer of the `LocationEncoder`. While these changes improved spatial frequency modeling, the **image_side** still relies on a generic pre-trained CLIP extractor. 

The model currently lacks sensitivity to domain-specific street-view features—such as regional architectural styles, specific road signage, and local vegetation. This stage aims to bridge the visual domain gap using **LoRA (Low-Rank Adaptation)** on the ImageEncoder.

## 2. Implementation Plan

### 2.1 LoRA Configuration
*   **Target Modules**: LoRA will be injected into the `vision_model` of the CLIP backbone, specifically targeting the **Self-Attention** blocks.
    *   **Targets**: `q_proj`, `v_proj`.
    *   **Layer Selection**: To accelerate training and focus on high-level semantics, LoRA adapters will be applied **only to the last 6 layers** (index 18 to 23) of the ViT-L/14 encoder.
*   **Hyperparameters**: 
    *   **Rank (r)**: 8 or 16.
    *   **Alpha**: 16 or 32 (typically $r \times 2$).
    *   **Dropout**: 0.05.

### 2.2 Training Strategy

**A. Trainable Parameters (requires_grad = True):**
1.  **LoRA Adapters**: The newly injected $A$ and $B$ matrices in the targeted ViT layers.
2.  **Image Projection Head**: The `self.mlp` in `ImageEncoder` (768 $\rightarrow$ 768 $\rightarrow$ 512).
3.  **SigmaSelector**: The image-conditioned attention module (takes image features + GPS, produces per-pair routing weights).
4.  **LocationEncoder Tail**: The last layer of the `LocationEncoderCapsule`.

**B. Frozen Weights:**
*   All original weights of the CLIP Vision Transformer (excluding the LoRA-injected parts).
*   The backbone layers of the LocationEncoder.

**C. Optimization & Efficiency Protocol:**
*   **Mixed-Precision Training**: Use `torch.cuda.amp` (FP16/BFloat16) to significantly decrease iteration time and memory usage.
*   **Disable Gradient Checkpointing**: Ensure `gradient_checkpointing` is disabled to maximize training speed (provided memory is sufficient).
*   **Low Learning Rate**: Use a base learning rate of $1 \times 10^{-4}$ for LoRA/MLP, and $5 \times 10^{-5}$ for location-related modules.
*   **Feasibility Check**: Perform a "Sanity Run" on a small subset (900 images for training, 100 for validation; refer to `scripts/sample_streetview_subset.py`) to verify gradient flow before the full 11K run.

## 3. Expected Result

### 3.1 Evaluation Metrics
Evaluation will be based on the **Great-Circle Distance** between predictions and ground-truth:
*   **Recall@k**: Success rate within thresholds of **1km, 25km, 200km, 750km, and 2500km**.
*   **Median Error (km)**: The median distance error across the test set.

### 3.2 Baseline Comparison
*   **Baseline**: Image-conditioned SigmaSelector + Unfrozen last layer of LocationEncoder.
*   **Target**: The LoRA-enhanced model is expected to achieve a significant improvement in fine-grained metrics (**Recall@1km** and **Recall@25km**), indicating that the model has effectively learned domain-specific street-view cues.

**NOTICE**: Baseline checkpoints (e.g., `baseline_v1.pth`) must be strictly version-controlled to ensure a rigorous comparison.