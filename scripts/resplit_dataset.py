#!/usr/bin/env python3
"""Re-split the street-view dataset with spatial stratification.

The original split (random.Random.shuffle) scatters images taken at the same GPS
coordinate across train/val/test.  For a panorama dataset where many images
share a location, this creates data leakage: the model can memorize GPS
coordinates from training images at the same spot.

This script groups images by their exact (latitude, longitude) coordinate,
shuffles the *groups* instead of individual images, then assigns groups to
train / val / test to approximate the target ratios (default 80/10/10).

Output: new train_subset.csv, val_subset.csv, test_subset.csv that replace the
originals.  The all_subset.csv and image directory are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALL_CSV = PROJECT_ROOT / "data/streetview_pano/all_subset.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/streetview_pano"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-split street-view dataset with GPS-based spatial stratification"
    )
    parser.add_argument("--all-csv", type=Path, default=DEFAULT_ALL_CSV,
                        help="Path to all_subset.csv containing every image")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory to write new train/val/test CSVs")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backup-suffix", type=str, default=".pre_resplit",
                        help="Suffix for backing up original CSVs (empty to skip)")
    return parser.parse_args()


def load_samples(all_csv: Path) -> list[tuple[str, float, float]]:
    """Read all_subset.csv, return list of (img_file, lat, lon)."""
    df = pd.read_csv(all_csv)
    samples = []
    for _, row in df.iterrows():
        samples.append((str(row["IMG_FILE"]), float(row["LAT"]), float(row["LON"])))
    return samples


def group_by_gps(samples: list[tuple[str, float, float]]) -> dict[tuple[float, float], list[str]]:
    """Group image filenames by their exact (lat, lon) coordinate."""
    groups: dict[tuple[float, float], list[str]] = defaultdict(list)
    for img_file, lat, lon in samples:
        groups[(lat, lon)].append(img_file)
    return groups


def assign_splits(
    groups: dict[tuple[float, float], list[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Assign groups to train/val/test to approximate target ratios.

    Shuffles groups, then greedily assigns each group to the split that is
    currently most under its target.  This keeps whole GPS-location groups
    together in one split.
    """
    rng = random.Random(seed)

    # Sort groups descending by size so large groups are placed first
    # (reduces ratio deviation from targets)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda x: len(x[1]), reverse=True)

    total = sum(len(imgs) for _, imgs in group_items)
    targets = {
        "train": int(total * train_ratio),
        "val": int(total * val_ratio),
        "test": total - int(total * train_ratio) - int(total * val_ratio),
    }

    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for _gps, images in group_items:
        # Pick the split that is most under its target proportion
        deficits = {
            split: (targets[split] - counts[split]) / max(targets[split], 1)
            for split in ["train", "val", "test"]
        }
        best_split = max(deficits, key=deficits.get)
        assigned[best_split].extend(images)
        counts[best_split] += len(images)

    return assigned


def write_split_csv(output_dir: Path, split_name: str, images: list[str],
                    all_samples: list[tuple[str, float, float]]) -> None:
    """Write a CSV with IMG_FILE, LAT, LON for the given split."""
    # Build lookup from img_file -> (lat, lon)
    lookup = {img: (lat, lon) for img, lat, lon in all_samples}

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split_name}_subset.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["IMG_FILE", "LAT", "LON"])
        writer.writeheader()
        for img in sorted(images):
            lat, lon = lookup[img]
            writer.writerow({"IMG_FILE": img, "LAT": lat, "LON": lon})


def main() -> int:
    args = parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0")

    if not args.all_csv.exists():
        raise FileNotFoundError(f"all_subset.csv not found: {args.all_csv}")

    # Backup original split CSVs
    if args.backup_suffix:
        for name in ("train", "val", "test"):
            csv_path = args.output_dir / f"{name}_subset.csv"
            if csv_path.exists():
                backup_path = csv_path.with_suffix(f"{args.backup_suffix}.csv")
                csv_path.rename(backup_path)
                print(f"Backed up: {csv_path} -> {backup_path}")

    # Load and group
    samples = load_samples(args.all_csv)
    print(f"Loaded {len(samples)} images from {args.all_csv}")

    groups = group_by_gps(samples)
    print(f"Unique GPS locations: {len(groups)}")
    print(f"Images per location: min={min(len(g) for g in groups.values())}, "
          f"max={max(len(g) for g in groups.values())}, "
          f"mean={sum(len(g) for g in groups.values()) / len(groups):.1f}")

    # Assign splits
    assigned = assign_splits(
        groups,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    for name in ("train", "val", "test"):
        imgs = assigned[name]
        actual_pct = len(imgs) / len(samples) * 100
        target_pct = getattr(args, f"{name}_ratio") * 100
        print(f"{name}: {len(imgs)} images ({actual_pct:.1f}%, target {target_pct:.1f}%)")

    # Write
    for name in ("train", "val", "test"):
        write_split_csv(args.output_dir, name, assigned[name], samples)

    print(f"\nNew split CSVs written to {args.output_dir}/")

    # Verify no cross-split GPS leakage
    train_gps = {gps for gps, imgs in groups.items()
                 if any(i in assigned["train"] for i in imgs)}
    val_gps = {gps for gps, imgs in groups.items()
               if any(i in assigned["val"] for i in imgs)}
    test_gps = {gps for gps, imgs in groups.items()
                if any(i in assigned["test"] for i in imgs)}

    tv = len(train_gps & val_gps)
    tt = len(train_gps & test_gps)
    vt = len(val_gps & test_gps)
    print(f"Cross-split GPS overlap: train∩val={tv}, train∩test={tt}, val∩test={vt}")
    if tv == 0 and tt == 0 and vt == 0:
        print("No GPS leakage between splits.")
    else:
        print("WARNING: some GPS coordinates still appear in multiple splits "
              "(this should not happen with exact-GPS grouping).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
