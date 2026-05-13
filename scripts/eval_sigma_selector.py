#!/usr/bin/env python3
"""Evaluate GeoCLIP baseline or SigmaSelector mode on configured datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from geoclip import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_val_transform
from geoclip.train.eval import eval_images


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GeoCLIP baseline or SigmaSelector variant")
    parser.add_argument("--dataset", choices=["im2gps3k", "streetview_pano"], required=True)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use-sigma-selector", type=str2bool, default=False)
    parser.add_argument("--selector-checkpoint", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_dataset_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "im2gps3k":
        return (
            Path("data/im2gps3k/test_subset.csv"),
            Path("data/im2gps3k/images/im2gps3ktest"),
        )

    return (
        Path("data/streetview_pano/test_subset.csv"),
        Path("data/streetview_pano/images"),
    )


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


def default_output_json(dataset: str, use_sigma_selector: bool) -> Path:
    suffix = "sigma_selector" if use_sigma_selector else "baseline"
    return Path(f"data/{dataset}/{suffix}_eval.json")


def detect_selector_variant_from_checkpoint(checkpoint_path: Path) -> str | None:
    """Read selector_variant from a training checkpoint if present."""
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        return payload.get("selector_variant", None)
    return None


def load_selector_checkpoint_if_needed(model: GeoCLIP, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Selector checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "location_encoder_state_dict" in payload:
        model.location_encoder.load_state_dict(payload["location_encoder_state_dict"], strict=True)
        return

    if isinstance(payload, dict) and "selector_state_dict" in payload:
        state = payload["selector_state_dict"]
    else:
        state = payload

    model.location_encoder.sigma_selector.load_state_dict(state)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    default_csv, default_image_dir = resolve_dataset_paths(args.dataset)
    test_csv = args.test_csv or default_csv
    image_dir = args.image_dir or default_image_dir

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    dataset = GeoDataLoader(str(test_csv), str(image_dir), transform=img_val_transform())
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check CSV and image directory")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_image_gps,
    )

    selector_variant = None
    if args.use_sigma_selector and args.selector_checkpoint is not None:
        selector_variant = detect_selector_variant_from_checkpoint(args.selector_checkpoint)

    model = GeoCLIP(
        from_pretrained=True,
        use_sigma_selector=args.use_sigma_selector,
        selector_variant=selector_variant,
    ).to(device)
    model.gps_gallery = model.gps_gallery.to(device)

    if args.use_sigma_selector:
        if args.selector_checkpoint is None:
            raise ValueError("--selector-checkpoint is required when --use-sigma-selector is true")
        load_selector_checkpoint_if_needed(model, args.selector_checkpoint)

    metrics = eval_images(dataloader, model, device=device)

    output_json = args.output_json or default_output_json(args.dataset, args.use_sigma_selector)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "test_csv": str(test_csv),
        "image_dir": str(image_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
        "use_sigma_selector": args.use_sigma_selector,
        "selector_checkpoint": str(args.selector_checkpoint) if args.selector_checkpoint else None,
        "metrics": metrics,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved evaluation JSON: {output_json}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
