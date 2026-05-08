# Demo Run Guide

How to run the GeoTX demo notebooks stably.

## Quick Start

```bash
# 1. Activate environment
conda activate geoclip

# 2. Verify model weights are downloaded (Git LFS)
cd /mnt/d/ML/project/geo-clip
git lfs pull

# 3. Launch Jupyter from the project root (important for imports)
cd demos
jupyter notebook
```

## Running Demo 1 (Interactive Prediction)

1. Open `demo1_interactive_prediction.ipynb`
2. Run cells **in order** (Cell 1 → 2 → 3 → 4)
3. Wait for cell 2 to finish loading the model (~5–10 seconds on GPU)
4. After cell 3 renders the upload widget, **restart the kernel and re-run all cells** if you had previously executed the notebook before the FileUpload fix was applied
5. Click the "Upload" button, select a street-view image
6. The heatmap and top-1 prediction will appear

## Known Issue: FileUpload TypeError (Fixed)

The original `on_upload` handler in cell 4 used a pattern that only works with
ipywidgets 7.x, where `FileUpload.value` was a dict keyed by filename.

In ipywidgets 8.x, `FileUpload.value` returns a **tuple of dicts**:

```python
# BROKEN (ipywidgets 7.x pattern):
fname = next(iter(uploaded))
img_bytes = uploaded[fname]['content']  # TypeError on ipywidgets 8.x

# CORRECT (ipywidgets 8.x pattern):
file_info = uploaded[0]
fname = file_info['name']
img_bytes = file_info['content']
```

If you encounter `TypeError: tuple indices must be integers or slices, not Bunch`,
restart the kernel and re-run all cells — the fix is already applied in the notebook.

## Environment Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10+ | |
| PyTorch | 2.x | CUDA recommended |
| ipywidgets | 8.x | FileUpload API changed from 7.x |
| matplotlib | 3.7+ | |
| cartopy | 0.22+ | Optional, enables map projections |

## Common Issues

**Model weights not found:** Run `git lfs pull` to download checkpoint files.
Check that `outputs/negative_sampling/full_threshold_*/neg_sampling_best.pth` exists.

**Kernel running old code:** If you ran the notebook before the fix, the kernel
still has the old `on_upload` definition. Restart the kernel and re-run all cells.

**CUDA out of memory:** The model + gallery uses ~2 GB GPU memory. Close other
notebook kernels if needed.

**Cartopy import error:** Install with `conda install -c conda-forge cartopy`.
The demo falls back to a simple scatter plot without cartopy.
