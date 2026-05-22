#!/usr/bin/env python3
"""Sweep negative-sampling distance thresholds on the validation set.

Loads the best v0.2 checkpoint (SigmaSelector + LoRA, trained *without* negative
sampling), then evaluates on the validation set with different geographic mask
thresholds H in {50, 100, 200, 400, 800} km.  Reports which H yields the best
validation accuracy at each distance threshold.

No training is performed — this is a pure evaluation sweep.  The resulting
threshold can then be used to configure train_negative_sampling.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geoclip import GeoCLIP
from geoclip.model.GeoCLIP import negative_sample_mask
from geoclip.train.dataloader import GeoDataLoader, img_val_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LORA_CKPT = PROJECT_ROOT / "outputs/lora/full_20260512T154049Z/lora_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep negative-sampling thresholds on validation set")
    parser.add_argument("--checkpoint", type=Path, default=LORA_CKPT,
                        help="v0.2 checkpoint (no negative sampling baked in)")
    parser.add_argument("--val-csv", type=Path,
                        default=PROJECT_ROOT / "data/streetview_pano/val_subset.csv")
    parser.add_argument("--image-dir", type=Path,
                        default=PROJECT_ROOT / "data/streetview_pano/images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[1.0, 25.0, 200.0, 750.0, 2500.0],
                        help="H values (km) to sweep")
    parser.add_argument("--output-json", type=Path,
                        default=PROJECT_ROOT / "outputs/neg_threshold_sweep.json")
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


def load_model(checkpoint_path: Path, device: str) -> GeoCLIP:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    lora_cfg = checkpoint.get("lora_config", {})
    selector_variant = checkpoint.get("selector_variant", None)
    use_lora = any("lora_" in k for k in state_dict.keys())
    use_sigma_selector = any("sigma_selector" in k for k in state_dict.keys())

    queue_size = _infer_queue_size(state_dict)

    model = GeoCLIP(
        from_pretrained=True,
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


def _infer_queue_size(state_dict: dict) -> int:
    q = state_dict.get("gps_queue")
    if isinstance(q, torch.Tensor) and q.ndim == 2:
        return int(q.shape[1])
    return 4096


@torch.no_grad()
def evaluate_with_threshold(model: GeoCLIP, dataloader: DataLoader,
                            device: str, threshold_km: float) -> dict[str, float]:
    """Evaluate model with negatives within threshold_km masked from the loss.

    Since we only need accuracy (not loss), we mask negatives from the logits
    before argmax to simulate what the model *would* have been trained with.
    """
    from geoclip.train.eval import eval_images

    # Standard evaluation with full gallery (no mask) but we also measure
    # the loss with the mask applied to assess training signal quality.
    metrics = eval_images(dataloader, model, device=device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0

    for images, gps in dataloader:
        images = images.to(device)
        gps = gps.to(device)
        batch_size = images.size(0)
        if batch_size == 0:
            continue

        gps_queue = model.get_gps_queue()
        gps_all = torch.cat([gps, gps_queue], dim=0)
        logits = model(images, gps_all)

        mask = negative_sample_mask(gps_all, batch_size, "threshold", threshold_km)
        logits_masked = logits.clone()
        logits_masked[mask] = -float("inf")

        targets = torch.arange(batch_size, device=device, dtype=torch.long)
        loss = criterion(logits_masked, targets)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    avg_loss = total_loss / total_count if total_count > 0 else float("inf")
    return {
        "acc_1_km": float(metrics.get("acc_1_km", 0)),
        "acc_25_km": float(metrics.get("acc_25_km", 0)),
        "acc_200_km": float(metrics.get("acc_200_km", 0)),
        "acc_750_km": float(metrics.get("acc_750_km", 0)),
        "acc_2500_km": float(metrics.get("acc_2500_km", 0)),
        "val_loss": float(avg_loss),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds: {args.thresholds} km")

    dataset = GeoDataLoader(str(args.val_csv), str(args.image_dir),
                            transform=img_val_transform())
    print(f"Validation samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_image_gps,
    )

    print("Loading model...")
    model = load_model(args.checkpoint, device)

    # Baseline without any mask
    print("\nEvaluating baseline (no mask)...")
    baseline = evaluate_with_threshold(model, dataloader, device, threshold_km=0.0)

    results = []
    print("\nSweeping thresholds:")
    for h in args.thresholds:
        print(f"  H = {h:.0f} km...")
        metrics = evaluate_with_threshold(model, dataloader, device, h)
        metrics["threshold_km"] = h
        results.append(metrics)

    # Print comparison
    print(f"\n{'H (km)':>8} {'1 km':>8} {'25 km':>8} {'200 km':>8} {'750 km':>8} {'2500 km':>8} {'Val Loss':>10}")
    print("-" * 70)
    print(f"{'none':>8} {baseline['acc_1_km']*100:>7.2f}% {baseline['acc_25_km']*100:>7.2f}% "
          f"{baseline['acc_200_km']*100:>7.2f}% {baseline['acc_750_km']*100:>7.2f}% "
          f"{baseline['acc_2500_km']*100:>7.2f}%")
    for r in results:
        print(f"{r['threshold_km']:>8.0f} {r['acc_1_km']*100:>7.2f}% {r['acc_25_km']*100:>7.2f}% "
              f"{r['acc_200_km']*100:>7.2f}% {r['acc_750_km']*100:>7.2f}% "
              f"{r['acc_2500_km']*100:>7.2f}% {r['val_loss']:>10.4f}")

    output = {
        "checkpoint": str(args.checkpoint),
        "val_csv": str(args.val_csv),
        "val_samples": len(dataset),
        "baseline_no_mask": baseline,
        "sweep": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
