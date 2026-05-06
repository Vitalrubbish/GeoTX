# GeoCLIP Enhancement on Optimized Negative Sampling

## Problem Statement
After implementing SigmaSelector and LoRA on ImageEncoder, the performance on streetview images has been enhanced. However, there still exists space for optimization. In traditional .Since we do not always require the model to locate images with the accuracy of `1km`, `25km` or `200km`, therefore the negative example selection can be optimized.

## Main Logic
If the distance between the actual location and the location of the negative example is small, we won't choose it as our negative example.

Specifically, you can choose one of the following algorithm, or you can design your own logic of negative example selection.

1. Choose the examples where distance is higher than some threshold `H`. Since we use a small-scale dataset, `H` can not be really small, maybe it's better for you to set `H` as `200-750km`.
2. Choose images with the Top-K longest distance in current negative example batch. 

## Implementation
You need to modify `geoclip/model/GeoCLIP.py`, changing the logic of selection of negative examples.

**Notice**: You have to reserve the baseline for comparison. To be specific, we need to run both the current baseline (SigmaSelector and LoRA) and your new model (SigmaSelector, LoRA and Negative Sampling) after implementation.

The training and evaluating logic of the new model is similar to baseline model. You can refer to `docs/dev/LoRA.md` to get the logic.