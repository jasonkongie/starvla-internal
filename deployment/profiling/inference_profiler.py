# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Per-component GPU inference latency profiler using CUDA Events.

Usage:
    profiler = InferenceProfiler()
    profiler.time_start("vlm_backbone")
    # ... GPU work ...
    profiler.time_end("vlm_backbone")
    profiler.log_to_wandb(step=0, prefix="inference")
    profiler.reset()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class InferenceProfiler:
    """Accumulates GPU latency samples per named tag using torch.cuda.Event pairs.

    CUDA events are recorded immediately at time_start/time_end; elapsed_time()
    is deferred until an explicit synchronize() inside _resolve(). This avoids
    stalling the GPU during the profiling loop.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, torch.cuda.Event] = {}
        self._event_pairs: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._resolved: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Timing API
    # ------------------------------------------------------------------

    def time_start(self, tag: str) -> None:
        """Record a CUDA start event for `tag`."""
        if not torch.cuda.is_available():
            raise RuntimeError("InferenceProfiler requires a CUDA device.")
        if tag in self._pending:
            logger.warning(
                "time_start(%s) called while a previous start is still pending; prior start discarded.", tag
            )
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._pending[tag] = evt

    def time_end(self, tag: str) -> None:
        """Record a CUDA end event for `tag` and store the pair for later resolution."""
        if tag not in self._pending:
            raise KeyError(
                f"time_end({tag!r}) called without a matching time_start(). "
                f"Active pending tags: {list(self._pending.keys())}"
            )
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._event_pairs[tag].append((self._pending.pop(tag), evt))

    # ------------------------------------------------------------------
    # Resolution and statistics
    # ------------------------------------------------------------------

    def _resolve(self) -> None:
        """Synchronize the CUDA stream and resolve all pending event pairs to ms floats."""
        torch.cuda.synchronize()
        for tag, pairs in list(self._event_pairs.items()):
            for start, end in pairs:
                self._resolved[tag].append(start.elapsed_time(end))
            self._event_pairs[tag].clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return mean/std/p50/p95/p99/count per tag (all times in ms)."""
        self._resolve()
        result: Dict[str, Dict[str, float]] = {}
        for tag, samples in self._resolved.items():
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            result[tag] = {
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "count": int(len(arr)),
            }
        return result

    def log_to_wandb(self, step: int, prefix: str = "inference") -> None:
        """Log per-tag statistics to wandb under `{prefix}/{tag}/{stat}`.

        Also logs a wandb.Histogram per tag for the full latency distribution.
        """
        import wandb

        self._resolve()
        metrics: Dict[str, object] = {}
        for tag, samples in self._resolved.items():
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            base = f"{prefix}/{tag}"
            metrics[f"{base}/mean_ms"] = float(np.mean(arr))
            metrics[f"{base}/std_ms"] = float(np.std(arr))
            metrics[f"{base}/p50_ms"] = float(np.percentile(arr, 50))
            metrics[f"{base}/p95_ms"] = float(np.percentile(arr, 95))
            metrics[f"{base}/p99_ms"] = float(np.percentile(arr, 99))
            metrics[f"{base}/count"] = int(len(arr))
            metrics[f"{base}/histogram_ms"] = wandb.Histogram(arr.tolist())

        if metrics:
            wandb.log(metrics, step=step)
            logger.info("Logged %d profiling metrics to wandb at step %d.", len(metrics), step)
        else:
            logger.warning("log_to_wandb called but no profiling samples have been collected.")

    def reset(self) -> None:
        """Clear all accumulated samples and pending events."""
        self._pending.clear()
        self._event_pairs.clear()
        self._resolved.clear()
