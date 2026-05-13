#!/usr/bin/env python3
"""Train only SigmaSelector while freezing all other GeoCLIP parameters."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoclip import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_train_transform, img_val_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoCLIP SigmaSelector only")
    parser.add_argument("--mode", choices=["feasibility", "full"], default="feasibility")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=Path("data/streetview_pano/images"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--queue-size", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sigma_selector"))
    parser.add_argument(
        "--unfreeze-capsule-head",
        action="store_true",
        help="Also unfreeze each LocationEncoderCapsule head (final linear layer)",
    )
    parser.add_argument("--selector-variant", type=str, default=None,
                        choices=[None, "v0.1"],
                        help="SigmaSelector variant: None (GPS-only) or v0.1 (image-conditioned)")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.train_csv is not None and args.val_csv is not None:
        return args.train_csv, args.val_csv

    if args.mode == "feasibility":
        return (
            Path("data/streetview_pano/feasibility/train_subset.csv"),
            Path("data/streetview_pano/feasibility/val_subset.csv"),
        )

    return (Path("data/streetview_pano/train_subset.csv"), Path("data/streetview_pano/val_subset.csv"))


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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_for_sigma_training(model: GeoCLIP, unfreeze_capsule_head: bool = False) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    if not hasattr(model.location_encoder, "sigma_selector"):
        raise RuntimeError("SigmaSelector is not available. Ensure use_sigma_selector=True")

    trainable_params: list[torch.nn.Parameter] = []

    for param in model.location_encoder.sigma_selector.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    if unfreeze_capsule_head:
        for i in range(model.location_encoder.n):
            capsule = model.location_encoder._modules[f"LocEnc{i}"]
            for param in capsule.head.parameters():
                param.requires_grad = True
                trainable_params.append(param)

    return trainable_params


def train_one_epoch(
    model: GeoCLIP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0
    running_count = 0

    bar = tqdm(dataloader, total=len(dataloader), desc="Train", leave=False)
    for images, gps in bar:
        images = images.to(device)
        gps = gps.to(device)

        batch_size = images.size(0)
        if batch_size == 0:
            continue

        optimizer.zero_grad()

        gps_queue = model.get_gps_queue()
        gps_all = torch.cat([gps, gps_queue], dim=0)
        model.dequeue_and_enqueue(gps)

        logits_img_gps = model(images, gps_all)
        targets = torch.arange(batch_size, device=device, dtype=torch.long)
        loss = criterion(logits_img_gps, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_count += batch_size
        bar.set_postfix(loss=f"{loss.item():.4f}")

    if running_count == 0:
        return float("nan")
    return running_loss / running_count


@torch.no_grad()
def validate_one_epoch(
    model: GeoCLIP,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    model.eval()
    running_loss = 0.0
    running_count = 0

    for images, gps in tqdm(dataloader, total=len(dataloader), desc="Val", leave=False):
        images = images.to(device)
        gps = gps.to(device)

        batch_size = images.size(0)
        if batch_size == 0:
            continue

        gps_queue = model.get_gps_queue()
        gps_all = torch.cat([gps, gps_queue], dim=0)

        logits_img_gps = model(images, gps_all)
        targets = torch.arange(batch_size, device=device, dtype=torch.long)
        loss = criterion(logits_img_gps, targets)

        running_loss += loss.item() * batch_size
        running_count += batch_size

    if running_count == 0:
        return float("nan")
    return running_loss / running_count


def maybe_plot_curves(output_dir: Path, train_losses: list[float], val_losses: list[float]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plotting because matplotlib is unavailable: {exc}")
        return None

    fig_path = output_dir / "loss_curve.png"
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SigmaSelector Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return fig_path


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    train_csv, val_csv = resolve_paths(args)

    if args.queue_size % args.batch_size != 0:
        raise ValueError("--queue-size must be divisible by --batch-size due queue implementation")

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Train/val CSV not found. train={train_csv}, val={val_csv}. "
            "For feasibility mode, run scripts/sample_streetview_subset.py first."
        )

    train_dataset = GeoDataLoader(str(train_csv), str(args.image_dir), transform=img_train_transform())
    val_dataset = GeoDataLoader(str(val_csv), str(args.image_dir), transform=img_val_transform())

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Empty dataset detected. Check CSV paths and image directory")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_image_gps,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_image_gps,
    )

    model = GeoCLIP(from_pretrained=True, queue_size=args.queue_size, use_sigma_selector=True,
                    selector_variant=args.selector_variant).to(device)
    model.gps_gallery = model.gps_gallery.to(device)

    trainable_params = freeze_for_sigma_training(
        model,
        unfreeze_capsule_head=args.unfreeze_capsule_head,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    run_name = f"{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    print(f"Device: {device}")
    print(f"Train CSV: {train_csv} ({len(train_dataset)} samples)")
    print(f"Val CSV: {val_csv} ({len(val_dataset)} samples)")
    print(f"Run dir: {run_dir}")
    print(f"Unfreeze capsule head: {args.unfreeze_capsule_head}")
    print(f"SigmaSelector variant: {args.selector_variant or 'GPS-only'}")
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_param_count}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        latest_ckpt = run_dir / "selector_latest.pth"
        torch.save(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "selector_state_dict": model.location_encoder.sigma_selector.state_dict(),
                "location_encoder_state_dict": model.location_encoder.state_dict(),
                "selector_variant": args.selector_variant,
            },
            latest_ckpt,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = run_dir / "selector_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "selector_state_dict": model.location_encoder.sigma_selector.state_dict(),
                    "location_encoder_state_dict": model.location_encoder.state_dict(),
                    "selector_variant": args.selector_variant,
                },
                best_ckpt,
            )

    fig_path = maybe_plot_curves(run_dir, train_losses, val_losses)

    log_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "device": device,
        "seed": args.seed,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "image_dir": str(args.image_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "queue_size": args.queue_size,
        "unfreeze_capsule_head": args.unfreeze_capsule_head,
        "selector_variant": args.selector_variant,
        "trainable_param_count": trainable_param_count,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "loss_curve": str(fig_path) if fig_path is not None else None,
        "checkpoints": {
            "latest": str(run_dir / "selector_latest.pth"),
            "best": str(run_dir / "selector_best.pth"),
        },
    }

    log_path = run_dir / "train_log.json"
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    print(f"Saved log: {log_path}")
    print(f"Saved best checkpoint: {run_dir / 'selector_best.pth'}")
    print(f"Saved latest checkpoint: {run_dir / 'selector_latest.pth'}")
    if fig_path is not None:
        print(f"Saved loss curve: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
