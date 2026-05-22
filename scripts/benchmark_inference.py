#!/usr/bin/env python3
"""Benchmark inference latency and memory for GeoTX model variants.

Times forward passes for:
  - GeoCLIP baseline (no SigmaSelector, no LoRA)
  - v0.1 (SigmaSelector only)
  - v0.2 (SigmaSelector + LoRA)
  - v0.3 (SigmaSelector + LoRA, same as v0.2 at inference)

Across gallery sizes: 1K, 10K, 100K GPS coordinates.
Reports wall-clock time and peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from geoclip import GeoCLIP

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Best checkpoints from full training runs (used for v0.1/v0.2/v0.3 variants)
LORA_CKPT = PROJECT_ROOT / "outputs/lora/full_seed42_20260522T105852Z/lora_best.pth"
SIGMA_CKPT = PROJECT_ROOT / "outputs/sigma_selector/full_seed42_20260522T095827Z/selector_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference latency and memory")
    parser.add_argument("--gallery-sizes", type=int, nargs="+",
                        default=[1024, 10000, 100000],
                        help="Gallery sizes to benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of query images per forward pass")
    parser.add_argument("--warmup-iters", type=int, default=3,
                        help="Warmup iterations before timing")
    parser.add_argument("--bench-iters", type=int, default=10,
                        help="Number of timed iterations for averaging")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-json", type=Path,
                        default=PROJECT_ROOT / "outputs/inference_benchmark.json")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def load_baseline_model(device: str) -> GeoCLIP:
    model = GeoCLIP(from_pretrained=True, queue_size=2048,
                    use_sigma_selector=False, use_lora=False).to(device)
    model.gps_gallery = model.gps_gallery.to(device)
    model.eval()
    return model


def load_v01_model(device: str) -> GeoCLIP:
    """Load v0.1 (SigmaSelector only) from its checkpoint."""
    checkpoint = torch.load(SIGMA_CKPT, map_location="cpu")
    model = GeoCLIP(from_pretrained=True, queue_size=2048,
                    use_sigma_selector=True, use_lora=False,
                    selector_variant="v0.1").to(device)
    model.gps_gallery = model.gps_gallery.to(device)
    model.location_encoder.load_state_dict(checkpoint["location_encoder_state_dict"], strict=False)
    model.eval()
    return model


def load_v02_model(device: str) -> GeoCLIP:
    """Load v0.2 (SigmaSelector + LoRA) from its checkpoint."""
    checkpoint = torch.load(LORA_CKPT, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    lora_cfg = checkpoint.get("lora_config", {})
    model = GeoCLIP(from_pretrained=True, queue_size=2048,
                    use_sigma_selector=True, use_lora=True,
                    lora_r=lora_cfg.get("r", 4),
                    lora_alpha=lora_cfg.get("alpha", 8),
                    lora_dropout=lora_cfg.get("dropout", 0.05),
                    selector_variant="v0.1").to(device)
    model.gps_gallery = model.gps_gallery.to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def generate_gallery(size: int, device: str) -> torch.Tensor:
    """Generate a synthetic GPS gallery of the given size uniformly over the sphere."""
    rng = torch.Generator(device="cpu").manual_seed(42)
    # Uniform on sphere: lat = asin(2*u - 1), lon = uniform(-180, 180)
    u = torch.rand(size, generator=rng)
    lats = torch.rad2deg(torch.asin(2 * u - 1))
    lons = torch.rand(size, generator=rng) * 360 - 180
    return torch.stack([lats, lons], dim=1).to(device)


def benchmark_model(model: GeoCLIP, gallery_sizes: list[int], batch_size: int,
                    warmup_iters: int, bench_iters: int, device: str,
                    model_name: str) -> list[dict]:
    results = []
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)

    for gallery_size in gallery_sizes:
        gallery = generate_gallery(gallery_size, device)

        # Warmup
        for _ in range(warmup_iters):
            _ = model(dummy_image, gallery)

        if device == "cuda":
            torch.cuda.synchronize()
            mem_before = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        # Timed runs
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = model(dummy_image, gallery)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / bench_iters

        if device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GiB
        else:
            peak_mem = 0.0

        result = {
            "model": model_name,
            "gallery_size": gallery_size,
            "batch_size": batch_size,
            "avg_time_ms": round(avg_time * 1000, 2),
            "peak_gpu_memory_gib": round(peak_mem, 2),
        }
        results.append(result)

        mem_str = f", peak mem: {peak_mem:.2f} GiB" if device == "cuda" else ""
        print(f"  {model_name} | gallery={gallery_size:>6} | "
              f"avg {avg_time*1000:>8.2f} ms{mem_str}")

    return results


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Gallery sizes: {args.gallery_sizes}")
    print(f"Warmup: {args.warmup_iters} iters, Benchmark: {args.bench_iters} iters\n")

    all_results = []

    # Baseline
    print("Loading GeoCLIP baseline (frozen, no SigmaSelector, no LoRA)...")
    baseline = load_baseline_model(device)
    all_results += benchmark_model(baseline, args.gallery_sizes, args.batch_size,
                                   args.warmup_iters, args.bench_iters, device, "GeoCLIP Baseline")
    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    # v0.1
    print("\nLoading v0.1 (SigmaSelector only)...")
    v01 = load_v01_model(device)
    all_results += benchmark_model(v01, args.gallery_sizes, args.batch_size,
                                   args.warmup_iters, args.bench_iters, device, "GeoTX v0.1")
    del v01
    if device == "cuda":
        torch.cuda.empty_cache()

    # v0.2
    print("\nLoading v0.2 (SigmaSelector + LoRA)...")
    v02 = load_v02_model(device)
    all_results += benchmark_model(v02, args.gallery_sizes, args.batch_size,
                                   args.warmup_iters, args.bench_iters, device, "GeoTX v0.2")
    del v02
    if device == "cuda":
        torch.cuda.empty_cache()

    # Summary
    print("\nSummary:")
    print(f"{'Model':<20} {'Gallery':>8} {'Time (ms)':>10} {'Mem (GiB)':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['model']:<20} {r['gallery_size']:>8} {r['avg_time_ms']:>10.2f} {r['peak_gpu_memory_gib']:>10.2f}")

    output = {
        "device": device,
        "gallery_sizes": args.gallery_sizes,
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
        "results": all_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
