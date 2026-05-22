#!/usr/bin/env python3
"""Multi-seed evaluation wrapper for GeoTX checkpoints.

Given a set of checkpoints from different training seeds, runs evaluation on
each and reports mean +/- std across seeds at each distance threshold.

If only one checkpoint is provided, reports single-run results with a caveat
about statistical significance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from geoclip import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_val_transform
from geoclip.train.eval import eval_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed evaluation wrapper")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                        help="One or more checkpoint .pth files from different seeds")
    parser.add_argument("--test-csv", type=Path,
                        default=PROJECT_ROOT / "data/streetview_pano/test_subset.csv")
    parser.add_argument("--image-dir", type=Path,
                        default=PROJECT_ROOT / "data/streetview_pano/images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-json", type=Path,
                        default=PROJECT_ROOT / "outputs/multi_seed_eval.json")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def collate_image_gps(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    gps = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return images, gps


def _infer_queue_size(state_dict: dict) -> int:
    q = state_dict.get("gps_queue")
    if isinstance(q, torch.Tensor) and q.ndim == 2:
        return int(q.shape[1])
    return 4096


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> GeoCLIP:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    queue_size = _infer_queue_size(state_dict)
    lora_cfg = checkpoint.get("lora_config", {})
    selector_variant = checkpoint.get("selector_variant", None)

    use_lora = any("lora_" in k for k in state_dict.keys())
    use_sigma_selector = any("sigma_selector" in k for k in state_dict.keys())

    model = GeoCLIP(
        from_pretrained=False,
        queue_size=queue_size,
        use_sigma_selector=use_sigma_selector,
        use_lora=use_lora,
        lora_r=lora_cfg.get("r", 4),
        lora_alpha=lora_cfg.get("alpha", 8),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        selector_variant=selector_variant,
    ).to(device)
    model.gps_gallery = model.gps_gallery.to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def evaluate_checkpoint(checkpoint_path: Path, dataloader: DataLoader,
                        device: str) -> dict[str, float]:
    print(f"  Evaluating: {checkpoint_path.name}")
    model = load_model_from_checkpoint(checkpoint_path, device)
    metrics = eval_images(dataloader, model, device=device)
    return {
        "acc_1_km": float(metrics.get("acc_1_km", 0)),
        "acc_25_km": float(metrics.get("acc_25_km", 0)),
        "acc_200_km": float(metrics.get("acc_200_km", 0)),
        "acc_750_km": float(metrics.get("acc_750_km", 0)),
        "acc_2500_km": float(metrics.get("acc_2500_km", 0)),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Test set: {args.test_csv}")
    print(f"Checkpoints ({len(args.checkpoints)}):")
    for cp in args.checkpoints:
        print(f"  {cp}")

    dataset = GeoDataLoader(str(args.test_csv), str(args.image_dir),
                            transform=img_val_transform())
    print(f"Test samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_image_gps,
    )

    results_per_checkpoint = []
    for cp in args.checkpoints:
        metrics = evaluate_checkpoint(cp, dataloader, device)
        results_per_checkpoint.append({"checkpoint": str(cp), **metrics})

    thresholds = ["acc_1_km", "acc_25_km", "acc_200_km", "acc_750_km", "acc_2500_km"]
    threshold_labels = ["1 km", "25 km", "200 km", "750 km", "2500 km"]

    agg = {}
    for t, label in zip(thresholds, threshold_labels):
        vals = [r[t] for r in results_per_checkpoint]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        agg[label] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }

    n_seeds = len(args.checkpoints)
    print(f"\nResults across {n_seeds} seed{'s' if n_seeds > 1 else ''}:")
    print(f"{'Threshold':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 42)
    for t, label in zip(thresholds, threshold_labels):
        a = agg[label]
        print(f"{label:<10} {a['mean']:>7.2f}% {a['std']:>7.2f}% {a['min']:>7.2f}% {a['max']:>7.2f}%")

    if n_seeds == 1:
        print("\nWARNING: Only one checkpoint provided. Results are from a single training "
              "run and do not reflect seed variance. Interpret fine-grained differences "
              "(especially at 1 km) with caution.")

    output = {
        "n_checkpoints": n_seeds,
        "test_csv": str(args.test_csv),
        "test_samples": len(dataset),
        "thresholds": agg,
        "per_checkpoint": results_per_checkpoint,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
