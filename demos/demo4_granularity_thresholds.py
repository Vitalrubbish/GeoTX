#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from geopy.distance import geodesic as GD
from tqdm import tqdm

_root = Path.cwd().parent if Path.cwd().name == 'demos' else Path.cwd()
sys.path.insert(0, str(_root))

from demos.demo_utils import (
    load_geotx_model, STREETVIEW_CSV, STREETVIEW_IMAGES, plot_comparison_map,
)
from geoclip.train.dataloader import img_val_transform

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = load_geotx_model(device)
gallery_np = model.gps_gallery.cpu().numpy()


# In[ ]:


# ── Evaluate on a subset to find examples at each error threshold ──
df_test = pd.read_csv(STREETVIEW_CSV).sample(n=300, random_state=42)
transform = img_val_transform()

predictions = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc='Predicting'):
    img_path = STREETVIEW_IMAGES / row['IMG_FILE']
    if not img_path.exists():
        continue
    
    gt_lat, gt_lon = float(row['LAT']), float(row['LON'])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor, model.gps_gallery)
        probs = logits.softmax(dim=-1).cpu().numpy().flatten()
    
    top1_idx = int(np.argmax(probs))
    pred_lat, pred_lon = gallery_np[top1_idx]
    error_km = GD((gt_lat, gt_lon), (pred_lat, pred_lon)).km
    
    predictions.append({
        'image_path': img_path,
        'image': img,
        'gt_lat': gt_lat,
        'gt_lon': gt_lon,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon,
        'error_km': error_km,
        'probs': probs,
    })

print(f'Evaluated {len(predictions)} images')


# In[ ]:


# ── Categorize predictions by error threshold ──
thresholds = [
    ('< 25 km (City/Street)', 0, 25),
    ('< 200 km (State/Region)', 25, 200),
    ('< 750 km (Country)', 200, 750),
    ('> 2500 km (Continent / Failure)', 2500, float('inf')),
]

selected = {}
for label, lo, hi in thresholds:
    candidates = [p for p in predictions if lo <= p['error_km'] < hi]
    if candidates:
        # Pick the example closest to the middle of the range (for representativeness)
        mid = (lo + min(hi, 10000)) / 2
        candidates.sort(key=lambda x: abs(x['error_km'] - mid))
        selected[label] = candidates[0]
        print(f'{label}: {len(candidates)} candidates, selected error={candidates[0]["error_km"]:.1f} km')
    else:
        print(f'{label}: NO candidates found')


# In[ ]:


# ── Comparative visualization ──
# Use simple scatter maps (works without cartopy)
n = len(selected)
fig, axes = plt.subplots(2, n, figsize=(n * 5.5, 11))

for idx, (label, pred) in enumerate(selected.items()):
    ax_img = axes[0, idx] if n > 1 else axes[0]
    ax_map = axes[1, idx] if n > 1 else axes[1]

    # Top row: input image
    ax_img.imshow(pred['image'])
    ax_img.set_title(f'{label}Error: {pred["error_km"]:.1f} km', fontsize=10)
    ax_img.axis('off')

    # Bottom row: zoomed comparison map
    margin = max(
        abs(pred['pred_lat'] - pred['gt_lat']),
        abs(pred['pred_lon'] - pred['gt_lon'])
    ) * 1.8 + 5
    ax_map.set_xlim(
        min(pred['gt_lon'], pred['pred_lon']) - margin,
        max(pred['gt_lon'], pred['pred_lon']) + margin
    )
    ax_map.set_ylim(
        min(pred['gt_lat'], pred['pred_lat']) - margin,
        max(pred['gt_lat'], pred['pred_lat']) + margin
    )
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.grid(True, alpha=0.3)

    # Scatter probability heatmap (downsampled)
    n_show = min(5000, len(pred['probs']))
    idx_show = np.random.RandomState(0).choice(len(pred['probs']), n_show, replace=False)
    gallery_np_local = model.gps_gallery.cpu().numpy()
    ax_map.scatter(
        gallery_np_local[idx_show, 1], gallery_np_local[idx_show, 0],
        c=pred['probs'][idx_show], cmap='hot', alpha=0.3, s=1, zorder=2
    )

    # Ground truth + prediction markers
    ax_map.scatter(pred['gt_lon'], pred['gt_lat'], color='green', marker='^',
                   s=120, edgecolors='black', linewidth=0.8, zorder=5, label='Ground Truth')
    ax_map.scatter(pred['pred_lon'], pred['pred_lat'], color='red', marker='*',
                   s=180, edgecolors='black', linewidth=0.8, zorder=5, label='Prediction')
    ax_map.plot(
        [pred['gt_lon'], pred['pred_lon']],
        [pred['gt_lat'], pred['pred_lat']],
        color='blue', linewidth=1.5, linestyle='--', zorder=4
    )
    ax_map.legend(loc='lower left', fontsize=8)
    ax_map.set_title(f'{label} ({pred["error_km"]:.1f} km)', fontsize=10)

fig.suptitle(
    'GeoTX Prediction Accuracy at Different Geographic Scales'
    'Green △ = Ground Truth | Red ★ = Prediction | Blue --- = Error distance',
    fontsize=14, y=1.02
)
fig.tight_layout()
plt.show()


# In[ ]:


# ── Summary: Error Distribution ──
errors = [p['error_km'] for p in predictions]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(np.clip(errors, 0, 5000), bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax.axvline(np.median(errors), color='red', linestyle='--', label=f'Median: {np.median(errors):.0f} km')
ax.axvline(np.mean(errors), color='green', linestyle='--', label=f'Mean: {np.mean(errors):.0f} km')
ax.set_xlabel('Error (km)')
ax.set_ylabel('Count')
ax.set_title(f'Prediction Error Distribution (n={len(errors)})')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Cumulative accuracy by distance threshold
ax = axes[1]
dists = np.linspace(0, 5000, 500)
acc = [np.mean(np.array(errors) <= d) for d in dists]
ax.plot(dists, acc, color='#e74c3c', linewidth=2)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
for thresh, label in [(25, '25 km'), (200, '200 km'), (750, '750 km'), (2500, '2500 km')]:
    a = np.mean(np.array(errors) <= thresh)
    ax.scatter([thresh], [a], zorder=5)
    ax.annotate(f'{label}{a:.1%}', (thresh, a), textcoords='offset points',
                xytext=(10, -15), fontsize=8)
ax.set_xlabel('Distance Threshold (km)')
ax.set_ylabel('Cumulative Accuracy')
ax.set_title('Cumulative Accuracy by Distance Threshold')
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

