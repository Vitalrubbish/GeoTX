#!/usr/bin/env python3
"""Evaluate a LoRA-trained GeoCLIP model on configured datasets."""

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
    parser = argparse.ArgumentParser(description="Evaluate LoRA-trained GeoCLIP model")
    parser.add_argument("--dataset", choices=["im2gps3k", "streetview_pano"], required=True)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to LoRA checkpoint (.pth) from train_lora.py")
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


def default_output_json(dataset: str, checkpoint: Path) -> Path:
    stem = checkpoint.stem
    return Path(f"data/{dataset}/lora_{stem}_eval.json")


def infer_queue_size_from_checkpoint(checkpoint_data: dict) -> int:
    state_dict = checkpoint_data.get("model_state_dict", {})
    gps_queue = state_dict.get("gps_queue")
    if isinstance(gps_queue, torch.Tensor) and gps_queue.ndim == 2:
        # GeoCLIP stores queue as shape (2, queue_size)
        return int(gps_queue.shape[1])
    return 4096


def filter_incompatible_state_dict(model: GeoCLIP, state_dict: dict) -> tuple[dict, list[str]]:
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped.append(
                f"{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}"
            )
            continue
        filtered_state[key] = value

    return filtered_state, skipped


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    default_csv, default_image_dir = resolve_dataset_paths(args.dataset)
    test_csv = args.test_csv or default_csv
    image_dir = args.image_dir or default_image_dir

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    lora_config = checkpoint_data.get("lora_config", {})

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

    queue_size = infer_queue_size_from_checkpoint(checkpoint_data)

    selector_variant = checkpoint_data.get("selector_variant", None)

    model = GeoCLIP(
        from_pretrained=True,
        queue_size=queue_size,
        use_sigma_selector=True,
        use_lora=True,
        lora_r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("alpha", 16),
        lora_dropout=lora_config.get("dropout", 0.05),
        selector_variant=selector_variant,
    ).to(device)
    model.gps_gallery = model.gps_gallery.to(device)

    filtered_state, skipped_keys = filter_incompatible_state_dict(
        model,
        checkpoint_data["model_state_dict"],
    )
    model.load_state_dict(filtered_state, strict=False)

    if skipped_keys:
        print("Warning: skipped incompatible checkpoint tensors:")
        for item in skipped_keys:
            print(f"  - {item}")

    metrics = eval_images(dataloader, model, device=device)

    output_json = args.output_json or default_output_json(args.dataset, args.checkpoint)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "test_csv": str(test_csv),
        "image_dir": str(image_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
        "queue_size": queue_size,
        "checkpoint": str(args.checkpoint),
        "lora_config": lora_config,
        "metrics": metrics,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved evaluation JSON: {output_json}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
