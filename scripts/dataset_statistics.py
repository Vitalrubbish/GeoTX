#!/usr/bin/env python3
"""Generate statistics for the street-view panorama dataset.

Produces a JSON report with:
  - Image count, unique GPS coordinates, approximate unique locations
  - Geographic distribution (lat/lon spread, continental presence)
  - Pairwise distance statistics
  - Train/val/test split overlap analysis (spatial leakage check)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = {
    "train": PROJECT_ROOT / "data/streetview_pano/train_subset.csv",
    "val": PROJECT_ROOT / "data/streetview_pano/val_subset.csv",
    "test": PROJECT_ROOT / "data/streetview_pano/test_subset.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate street-view dataset statistics")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_SPLITS["val"])
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--output-json", type=Path,
                        default=PROJECT_ROOT / "outputs/dataset_statistics.json")
    parser.add_argument("--cluster-radius-km", type=float, default=1.0,
                        help="Radius for clustering nearby coordinates into unique locations")
    parser.add_argument("--spatial-leak-check-km", type=float, default=1.0,
                        help="Distance threshold for flagging train/test location overlap")
    return parser.parse_args()


def load_coords(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df[["LAT", "LON"]].values


def haversine_km(lat_a, lon_a, lat_b, lon_b):
    """Compute haversine distance(s) in km between point a and point(s) b.

    Args:
        lat_a, lon_a: Scalars (single point).
        lat_b, lon_b: Scalars or arrays (query point(s)).

    Returns:
        Scalar or array of distances in km.
    """
    R = 6371.0
    lat_a, lon_a = np.radians(float(lat_a)), np.radians(float(lon_a))
    lat_b, lon_b = np.radians(np.asarray(lat_b, dtype=float)), np.radians(np.asarray(lon_b, dtype=float))
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return R * 2 * np.arcsin(np.sqrt(a))


def count_unique_locations(coords: np.ndarray, cluster_radius_km: float = 1.0) -> int:
    """Approximate unique locations by counting coordinates > cluster_radius_km apart."""
    if len(coords) <= 1:
        return len(coords)
    accepted = [coords[0]]
    for pt in coords[1:]:
        dists = haversine_km(pt[0], pt[1],
                             np.array([a[0] for a in accepted]),
                             np.array([a[1] for a in accepted]))
        if np.all(dists > cluster_radius_km):
            accepted.append(pt)
    return len(accepted)


def geographic_summary(coords: np.ndarray) -> dict:
    lats, lons = coords[:, 0], coords[:, 1]
    return {
        "lat_range": [float(lats.min()), float(lats.max())],
        "lon_range": [float(lons.min()), float(lons.max())],
        "lat_mean": float(lats.mean()),
        "lat_std": float(lats.std()),
        "lon_mean": float(lons.mean()),
        "lon_std": float(lons.std()),
        "northern_hemisphere_pct": float((lats > 0).mean() * 100),
        "southern_hemisphere_pct": float((lats < 0).mean() * 100),
    }


def pairwise_distance_stats(coords: np.ndarray, max_samples: int = 500) -> dict:
    """Estimate pairwise distance statistics via sampling."""
    n = len(coords)
    rng = np.random.default_rng(42)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        coords = coords[idx]
        n = max_samples

    lats = coords[:, 0]
    lons = coords[:, 1]
    # Vectorized upper-triangle pairwise distances
    lat_i = lats[:, np.newaxis]
    lon_i = lons[:, np.newaxis]
    lat_j = lats[np.newaxis, :]
    lon_j = lons[np.newaxis, :]
    dlat = np.radians(lat_i - lat_j)
    dlon = np.radians(lon_i - lon_j)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat_i)) * np.cos(np.radians(lat_j)) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    dists = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    # Take upper triangle (excluding diagonal)
    triu = dists[np.triu_indices(n, k=1)]

    return {
        "sampled_pairs": len(triu),
        "min_km": float(triu.min()),
        "max_km": float(triu.max()),
        "mean_km": float(triu.mean()),
        "median_km": float(np.median(triu)),
        "std_km": float(triu.std()),
    }


def spatial_leakage_check(train_coords: np.ndarray, test_coords: np.ndarray,
                          threshold_km: float = 1.0) -> dict:
    """Check how many test points have at least one train point within threshold_km."""
    close_count = 0
    for test_pt in test_coords:
        dists = haversine_km(test_pt[0], test_pt[1],
                             train_coords[:, 0], train_coords[:, 1])
        if np.any(dists < threshold_km):
            close_count += 1
    return {
        "threshold_km": threshold_km,
        "test_points_with_nearby_train": close_count,
        "test_total": len(test_coords),
        "leakage_pct": float(close_count / len(test_coords) * 100),
    }


def main() -> int:
    args = parse_args()

    train = load_coords(args.train_csv)
    val = load_coords(args.val_csv)
    test = load_coords(args.test_csv)
    all_coords = np.concatenate([train, val, test], axis=0)

    stats = {
        "total_images": len(all_coords),
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "unique_gps_coordinates": int(len(np.unique(all_coords, axis=0))),
        "approximate_unique_locations": {
            f"cluster_{args.cluster_radius_km}km": count_unique_locations(
                all_coords, args.cluster_radius_km
            ),
        },
        "geographic_distribution": geographic_summary(all_coords),
        "pairwise_distance": pairwise_distance_stats(all_coords),
        "split_geographic_distribution": {
            "train": geographic_summary(train),
            "val": geographic_summary(val),
            "test": geographic_summary(test),
        },
        "spatial_leakage_train_to_test": spatial_leakage_check(
            train, test, args.spatial_leak_check_km
        ),
        "spatial_leakage_train_to_val": spatial_leakage_check(
            train, val, args.spatial_leak_check_km
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    print(f"Dataset statistics ({len(all_coords)} images):")
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"  Unique GPS coords: {stats['unique_gps_coordinates']}")
    locs = stats['approximate_unique_locations'][f"cluster_{args.cluster_radius_km}km"]
    print(f"  Approx unique locations (>{args.cluster_radius_km}km apart): {locs}")
    geo = stats["geographic_distribution"]
    print(f"  Lat range: [{geo['lat_range'][0]:.2f}, {geo['lat_range'][1]:.2f}]")
    print(f"  Lon range: [{geo['lon_range'][0]:.2f}, {geo['lon_range'][1]:.2f}]")
    dist = stats["pairwise_distance"]
    print(f"  Pairwise distance: median={dist['median_km']:.1f} km, max={dist['max_km']:.1f} km")
    leak = stats["spatial_leakage_train_to_test"]
    print(f"  Spatial leakage (test): {leak['leakage_pct']:.1f}% of test points "
          f"have a train point within {args.spatial_leak_check_km} km")
    print(f"Saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
