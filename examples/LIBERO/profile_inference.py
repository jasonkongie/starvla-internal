# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Standalone inference latency profiling for QwenGR00T on LIBERO-style inputs.

Does NOT require the LIBERO environment — uses synthetic images and instructions.
Requires a trained checkpoint (--ckpt_path).

Usage:
    python examples/LIBERO/profile_inference.py \
        --ckpt_path playground/Checkpoints/starvla/checkpoints/step-100000 \
        --num_warmup 10 \
        --num_profile 100 \
        --batch_size 1 \
        --wandb_project llavavla \
        --wandb_entity <your_entity> \
        --wandb_run_name profile_qwengr00t_bs1

    # Print-only (no wandb):
    python examples/LIBERO/profile_inference.py \
        --ckpt_path <ckpt> --num_warmup 5 --num_profile 20 --no_wandb
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_workspace_root = Path(__file__).parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from starVLA.model.framework.base_framework import baseframework
from deployment.profiling.inference_profiler import InferenceProfiler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s | %(message)s",
    datefmt="%m/%d [%H:%M:%S]",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile per-component inference latency for QwenGR00T.")
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to trained checkpoint directory (contains config.yaml + weights).",
    )
    p.add_argument("--num_warmup", type=int, default=10, help="Warmup iterations (not profiled).")
    p.add_argument("--num_profile", type=int, default=100, help="Profiled iterations.")
    p.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
    p.add_argument("--image_size", type=int, default=224, help="Synthetic image height/width (px).")
    p.add_argument("--state_dim", type=int, default=7, help="Robot state dimension.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wandb_project", type=str, default="llavavla")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="profile_qwengr00t")
    p.add_argument("--no_wandb", action="store_true", help="Skip wandb; print summary only.")
    return p.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_synthetic_batch(batch_size: int, image_size: int, state_dim: int) -> list[dict]:
    """Synthetic LIBERO-like observations (random pixels, fixed instruction)."""
    instruction = "pick up the red bowl and place it on the wooden surface"
    batch = []
    for _ in range(batch_size):
        arr = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        state = np.random.randn(1, state_dim).astype(np.float32)
        batch.append({"image": [img], "lang": instruction, "state": state})
    return batch


def gpu_memory_stats(device: str = "cuda") -> dict:
    if not torch.cuda.is_available():
        return {}
    idx = torch.cuda.current_device()
    return {
        "gpu_memory/allocated_mb": torch.cuda.memory_allocated(idx) / 1024**2,
        "gpu_memory/reserved_mb": torch.cuda.memory_reserved(idx) / 1024**2,
        "gpu_memory/peak_allocated_mb": torch.cuda.max_memory_allocated(idx) / 1024**2,
        "gpu_memory/peak_reserved_mb": torch.cuda.max_memory_reserved(idx) / 1024**2,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_profiling(args: argparse.Namespace) -> None:
    # 1. Load model
    logger.info("Loading model from: %s", args.ckpt_path)
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(args.device).eval()
    logger.info("Model loaded.")

    # 2. Synthetic batch (fixed across iterations for reproducible timing)
    batch = make_synthetic_batch(args.batch_size, args.image_size, args.state_dim)

    # 3. wandb init
    if not args.no_wandb:
        import wandb
        wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group="inference-profiling",
            config={
                "ckpt_path": args.ckpt_path,
                "num_warmup": args.num_warmup,
                "num_profile": args.num_profile,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "state_dim": args.state_dim,
                "device": args.device,
            },
        )

    profiler = InferenceProfiler()

    # 4. Warmup (no profiler — lets GPU caches and JIT settle)
    logger.info("Warming up (%d iters)...", args.num_warmup)
    for _ in range(args.num_warmup):
        model.predict_action(batch)
    torch.cuda.synchronize()

    # 5. Reset peak memory stats after warmup for clean peak readings
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 6. Profiled iterations
    logger.info("Profiling (%d iters)...", args.num_profile)
    for _ in range(args.num_profile):
        model.predict_action(batch, profiler=profiler)

    # 7. Print summary
    stats = profiler.summary()
    logger.info("=" * 68)
    logger.info("  Inference Latency Summary  (batch_size=%d, device=%s)", args.batch_size, args.device)
    logger.info("  %-16s %8s %8s %8s %8s %8s", "component", "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms")
    logger.info("  " + "-" * 62)
    for tag in ("total", "preprocessing", "vlm_backbone", "action_head"):
        if tag not in stats:
            continue
        s = stats[tag]
        logger.info(
            "  %-16s %8.2f %8.2f %8.2f %8.2f %8.2f",
            tag, s["mean_ms"], s["std_ms"], s["p50_ms"], s["p95_ms"], s["p99_ms"],
        )
    logger.info("=" * 68)

    # 8. Log to wandb
    if not args.no_wandb:
        import wandb
        profiler.log_to_wandb(step=0, prefix="inference")
        mem = gpu_memory_stats(args.device)
        if mem:
            wandb.log(mem, step=0)
        wandb.finish()
        logger.info("Results logged to wandb project '%s', run '%s'.", args.wandb_project, args.wandb_run_name)


if __name__ == "__main__":
    run_profiling(parse_args())
