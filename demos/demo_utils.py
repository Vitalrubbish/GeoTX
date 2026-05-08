"""Shared utilities for GeoTX demo notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from PIL import Image
from geopy.distance import geodesic as GD

# Ensure project root is on sys.path for geoclip imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from geoclip import GeoCLIP
from geoclip.train.dataloader import img_val_transform

# ── Paths ──────────────────────────────────────────────────────────────────
WEIGHTS_DIR = _project_root / "geoclip" / "model" / "weights"
GPS_GALLERY_CSV = _project_root / "geoclip" / "model" / "gps_gallery" / "coordinates_100K.csv"
STREETVIEW_CSV = _project_root / "data" / "streetview_pano" / "all_subset.csv"
STREETVIEW_IMAGES = _project_root / "data" / "streetview_pano" / "images"

# Best checkpoints from full training runs
LORA_CKPT = _project_root / "outputs" / "lora" / "full_20260430T061933Z" / "lora_best.pth"
SIGMA_CKPT = _project_root / "outputs" / "sigma_selector" / "full_20260427T122125Z" / "selector_best.pth"
NEG_SAMPLING_CKPT = _project_root / "outputs" / "negative_sampling" / "full_threshold_20260506T053749Z" / "neg_sampling_best.pth"


def load_geotx_model(device: str | None = None) -> GeoCLIP:
    """Load GeoTX model with LoRA + SigmaSelector + Negative Sampling weights.

    Uses the negative-sampling checkpoint (which includes LoRA + SigmaSelector
    weights from prior training stages).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(NEG_SAMPLING_CKPT, map_location="cpu")
    queue_size = _infer_queue_size(checkpoint)
    lora_cfg = checkpoint.get("lora_config", {})

    model = GeoCLIP(
        from_pretrained=True,
        queue_size=queue_size,
        use_sigma_selector=True,
        use_lora=True,
        lora_r=lora_cfg.get("r", 4),
        lora_alpha=lora_cfg.get("alpha", 8),
        lora_dropout=lora_cfg.get("dropout", 0.05),
    ).to(device)
    model.gps_gallery = model.gps_gallery.to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"GeoTX model loaded on {device} (queue_size={queue_size}, "
          f"lora_r={lora_cfg.get('r', 4)}, lora_alpha={lora_cfg.get('alpha', 8)})")
    return model


def _infer_queue_size(checkpoint: dict) -> int:
    state = checkpoint.get("model_state_dict", checkpoint)
    q = state.get("gps_queue")
    if isinstance(q, torch.Tensor) and q.ndim == 2:
        return int(q.shape[1])
    return 4096


def load_image(image_path: str | Path, device: str = "cpu") -> torch.Tensor:
    """Load and preprocess a single image for the model."""
    image = Image.open(image_path).convert("RGB")
    transform = img_val_transform()
    return transform(image).unsqueeze(0).to(device)


def load_gps_gallery() -> torch.Tensor:
    """Load the 100K GPS gallery coordinates."""
    df = pd.read_csv(GPS_GALLERY_CSV)
    return torch.tensor(df[["LAT", "LON"]].values, dtype=torch.float32)


def load_dataset_coordinates(csv_path: str | Path | None = None) -> np.ndarray:
    """Load lat/lon pairs from the streetview dataset CSV."""
    path = Path(csv_path) if csv_path else STREETVIEW_CSV
    df = pd.read_csv(path)
    return df[["LAT", "LON"]].values


# ── Map plotting ───────────────────────────────────────────────────────────

def plot_world_heatmap(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    ax=None,
    title: str = "",
    cmap: str = "hot",
    marker_lat: float | None = None,
    marker_lon: float | None = None,
    marker_label: str = "Prediction",
    marker_color: str = "red",
    marker_size: int = 120,
    alpha: float = 0.6,
    s: float = 1.0,
):
    """Plot a global probability/density heatmap on a Mercator projection.

    Args:
        lats, lons: Coordinates for each point.
        values: Color-mapped values (probabilities, densities, etc.).
        ax: Optional matplotlib axes.
        title: Plot title.
        cmap: Colormap name.
        marker_lat, marker_lon: Optional marker position.
        marker_label, marker_color, marker_size: Marker styling.
        alpha: Scatter alpha.
        s: Scatter point size.
    """
    import matplotlib.pyplot as plt

    try:
        _plot_with_cartopy(
            lats, lons, values, ax, title, cmap,
            marker_lat, marker_lon, marker_label, marker_color, marker_size,
            alpha, s,
        )
    except ImportError:
        _plot_fallback(
            lats, lons, values, ax, title, cmap,
            marker_lat, marker_lon, marker_label, marker_color, marker_size,
            alpha, s,
        )


def _plot_with_cartopy(
    lats, lons, values, ax, title, cmap,
    marker_lat, marker_lon, marker_label, marker_color, marker_size,
    alpha, s,
):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(14, 7),
            subplot_kw={"projection": ccrs.Mercator()},
        )
    else:
        fig = ax.figure

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d0e0f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=1)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    sc = ax.scatter(
        lons, lats,
        c=values, cmap=cmap,
        alpha=alpha, s=s,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    if marker_lat is not None and marker_lon is not None:
        ax.scatter(
            marker_lon, marker_lat,
            color=marker_color, marker="*", s=marker_size,
            edgecolors="black", linewidth=0.8,
            transform=ccrs.PlateCarree(),
            zorder=5, label=marker_label,
        )
        ax.legend(loc="lower left", fontsize=10)

    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig, ax


def _plot_fallback(
    lats, lons, values, ax, title, cmap,
    marker_lat, marker_lon, marker_label, marker_color, marker_size,
    alpha, s,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    else:
        fig = ax.figure

    # Draw simple world outline via scatter of landmass points isn't great;
    # just use a blank axes with grid.
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    sc = ax.scatter(
        lons, lats,
        c=values, cmap=cmap,
        alpha=alpha, s=s,
    )

    if marker_lat is not None and marker_lon is not None:
        ax.scatter(
            marker_lon, marker_lat,
            color=marker_color, marker="*", s=marker_size,
            edgecolors="black", linewidth=0.8,
            zorder=5, label=marker_label,
        )
        ax.legend(loc="lower left", fontsize=10)

    plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig, ax


def plot_comparison_map(
    gt_lat: float,
    gt_lon: float,
    pred_lat: float,
    pred_lon: float,
    heatmap_lats: np.ndarray | None = None,
    heatmap_lons: np.ndarray | None = None,
    heatmap_vals: np.ndarray | None = None,
    title: str = "",
):
    """Plot a map with ground truth (green), prediction (red), error line, and optional heatmap."""
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _has_cartopy = True
    except ImportError:
        _has_cartopy = False

    error_km = GD((gt_lat, gt_lon), (pred_lat, pred_lon)).km

    if _has_cartopy:
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={"projection": ccrs.Mercator()},
        )
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#d0e0f0", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=1)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        if heatmap_lats is not None:
            ax.scatter(
                heatmap_lons, heatmap_lats, c=heatmap_vals,
                cmap="hot", alpha=0.4, s=0.5,
                transform=ccrs.PlateCarree(), zorder=2,
            )

        ax.scatter(gt_lon, gt_lat, color="green", marker="^", s=100,
                   edgecolors="black", linewidth=0.8,
                   transform=ccrs.PlateCarree(), zorder=5, label="Ground Truth")
        ax.scatter(pred_lon, pred_lat, color="red", marker="*", s=150,
                   edgecolors="black", linewidth=0.8,
                   transform=ccrs.PlateCarree(), zorder=5, label="Prediction")
        ax.plot([gt_lon, pred_lon], [gt_lat, pred_lat],
                color="blue", linewidth=1.5, linestyle="--",
                transform=ccrs.Geodetic(), zorder=4)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        margin = max(abs(pred_lat - gt_lat), abs(pred_lon - gt_lon)) * 1.5 + 2
        ax.set_xlim(min(gt_lon, pred_lon) - margin, max(gt_lon, pred_lon) + margin)
        ax.set_ylim(min(gt_lat, pred_lat) - margin, max(gt_lat, pred_lat) + margin)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)

        if heatmap_lats is not None:
            ax.scatter(heatmap_lons, heatmap_lats, c=heatmap_vals,
                       cmap="hot", alpha=0.4, s=0.5)

        ax.scatter(gt_lon, gt_lat, color="green", marker="^", s=100,
                   edgecolors="black", linewidth=0.8, zorder=5, label="Ground Truth")
        ax.scatter(pred_lon, pred_lat, color="red", marker="*", s=150,
                   edgecolors="black", linewidth=0.8, zorder=5, label="Prediction")
        ax.plot([gt_lon, pred_lon], [gt_lat, pred_lat],
                color="blue", linewidth=1.5, linestyle="--", zorder=4)

    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"{title}\nError: {error_km:.1f} km", fontsize=12)
    fig.tight_layout()
    return fig, ax, error_km
