#!/usr/bin/env python3
"""Benchmark the training data pipeline to measure GPU utilization and bottlenecks.

Measures per-step timing for 5 pipeline configurations:

  eager_local       — pre-decoded numpy arrays in memory (zero fetch overhead)
  lazy_local_sync   — LazyColumn decode per batch, synchronous on main thread
  lazy_local_pre2   — LazyColumn with prefetch_size=2 background thread
  lazy_local_pre4   — LazyColumn with prefetch_size=4 background thread
  lazy_ray          — RayDataset + _with_lazy_decode (distributed backend)

Per step, records:
  t_fetch  — time next_batch() blocks  (= GPU idle time without prefetch)
  t_gpu    — time simulated GPU work takes
  util_pct — GPU utilization = t_gpu / (t_fetch + t_gpu) × 100

GPU work is simulated by time.sleep(--gpu-work-ms / 1000), which gives precise
control over the fetch/GPU ratio independent of hardware.  Vary --gpu-work-ms
to understand the regime you care about:

  --gpu-work-ms 5    small model / fast GPU   → decode usually dominates
  --gpu-work-ms 30   mid-size model            → decode can dominate
  --gpu-work-ms 100  large model / slow GPU   → GPU dominates, decode hidden

Run:
    python scripts/benchmark_training_pipeline.py [options]

Requirements: torchaudio (for WAV generation), ray[data] (for lazy_ray mode).
"""

import argparse
import os
import sys
import tempfile
import time
from statistics import mean, median, quantiles

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: synthetic WAV files + metadata
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_RATE = 16_000
_DURATION_S = 0.5


def _write_wav_files(dest_dir: str, n: int) -> list[str]:
    import torch
    import torchaudio

    os.makedirs(dest_dir, exist_ok=True)
    n_samples = int(_DURATION_S * _SAMPLE_RATE)
    silence = torch.zeros(1, n_samples)
    paths = []
    for i in range(n):
        p = os.path.join(dest_dir, f"audio_{i:05d}.wav")
        torchaudio.save(p, silence, _SAMPLE_RATE)
        paths.append(p)
    return paths


def _lazy_audio_metadata(feature_dim: int = 8, max_length: int = 23) -> dict:
    return {
        "lazy": True,
        "reshape": None,
        "lazy_audio_params": {
            "audio_feature_dict": {
                "type": "fbank",
                "window_length_in_s": 0.04,
                "window_shift_in_s": 0.02,
                "num_filter_bands": feature_dim,
            },
            "feature_dim": feature_dim,
            "max_length": max_length,
            "padding_value": 0.0,
            "normalization_type": None,
        },
    }


def _make_decode_fn(feature_dim: int, max_length: int):
    from ludwig.features.audio_feature import AudioFeatureMixin

    p = _lazy_audio_metadata(feature_dim, max_length)["lazy_audio_params"]
    return AudioFeatureMixin._make_lazy_decode_fn(
        audio_feature_dict=p["audio_feature_dict"],
        feature_dim=p["feature_dim"],
        max_length=p["max_length"],
        padding_value=p["padding_value"],
        normalization_type=p["normalization_type"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timing loop
# ─────────────────────────────────────────────────────────────────────────────

StepTiming = dict  # {'t_fetch': float, 't_gpu': float, 'epoch': int, 'step': int}


def _run_timing_loop(batcher, n_epochs: int, gpu_work_s: float, n_warmup: int) -> list[StepTiming]:
    """Run the batcher for n_epochs and return per-step timings after warmup."""
    timings = []
    global_step = 0

    for epoch in range(n_epochs):
        batcher.set_epoch(epoch, batcher.batch_size)
        while not batcher.last_batch():
            t0 = time.perf_counter()
            _batch = batcher.next_batch()
            t1 = time.perf_counter()

            # Simulate GPU forward + backward pass
            time.sleep(gpu_work_s)

            t2 = time.perf_counter()

            if global_step >= n_warmup:
                timings.append(
                    {
                        "t_fetch": t1 - t0,
                        "t_gpu": t2 - t1,
                        "epoch": epoch,
                        "step": global_step,
                    }
                )
            global_step += 1

    return timings


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    qs = quantiles(values, n=100) if len(values) >= 2 else [values[0]] * 99
    return {
        "mean": mean(values),
        "p50": median(values),
        "p95": qs[94] if len(qs) > 94 else max(values),
        "p99": qs[98] if len(qs) > 98 else max(values),
        "min": min(values),
        "max": max(values),
    }


def _analyze(timings: list[StepTiming], n_samples: int, n_epochs: int) -> dict:
    if not timings:
        return {}
    fetch_ms = [t["t_fetch"] * 1000 for t in timings]
    gpu_ms = [t["t_gpu"] * 1000 for t in timings]
    util = [g / (f + g) * 100 for f, g in zip(fetch_ms, gpu_ms) if (f + g) > 0]
    total_s = sum(t["t_fetch"] + t["t_gpu"] for t in timings)
    sps = (n_samples * n_epochs) / total_s if total_s > 0 else 0
    return {
        "fetch_ms": _stats(fetch_ms),
        "gpu_ms": _stats(gpu_ms),
        "util_pct": _stats(util),
        "sps": sps,
        "n_steps": len(timings),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mode: eager_local (pre-decoded arrays)
# ─────────────────────────────────────────────────────────────────────────────


def bench_eager_local(paths, batch_size, epochs, feature_dim, max_length, gpu_work_s, n_warmup):
    from concurrent.futures import ThreadPoolExecutor

    from ludwig.data.dataset.pandas import PandasDataset

    decode_fn = _make_decode_fn(feature_dim, max_length)
    with ThreadPoolExecutor(max_workers=min(16, len(paths))) as ex:
        decoded = np.stack(list(ex.map(decode_fn, paths)))

    proc_col = "audio_proc"
    feature_name = "audio_0"
    features = {proc_col: {"name": feature_name, "column": feature_name, "type": "audio"}}
    # Non-lazy metadata: decoded array stored directly
    training_set_metadata = {feature_name: {"lazy": False, "reshape": (feature_dim, max_length)}}
    ds = PandasDataset(
        {proc_col: decoded.reshape(len(paths), -1)},
        features,
        data_cache_fp=None,
        training_set_metadata=training_set_metadata,
    )
    with ds.initialize_batcher(batch_size=batch_size, should_shuffle=False) as batcher:
        return _run_timing_loop(batcher, epochs, gpu_work_s, n_warmup)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: lazy_local (sync or prefetch)
# ─────────────────────────────────────────────────────────────────────────────


def bench_lazy_local(paths, batch_size, epochs, feature_dim, max_length, gpu_work_s, n_warmup, prefetch_size=0):

    from ludwig.data.dataset.pandas import PandasDataset

    proc_col = "audio_proc"
    feature_name = "audio_0"
    features = {proc_col: {"name": feature_name, "column": feature_name, "type": "audio"}}
    training_set_metadata = {feature_name: _lazy_audio_metadata(feature_dim, max_length)}
    ds = PandasDataset(
        {proc_col: np.array(paths, dtype=object)},
        features,
        data_cache_fp=None,
        training_set_metadata=training_set_metadata,
    )
    with ds.initialize_batcher(batch_size=batch_size, should_shuffle=False, prefetch_size=prefetch_size) as batcher:
        return _run_timing_loop(batcher, epochs, gpu_work_s, n_warmup)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: lazy_ray
# ─────────────────────────────────────────────────────────────────────────────


def bench_lazy_ray(paths, batch_size, epochs, feature_dim, max_length, gpu_work_s, n_warmup):
    import pandas as pd
    import ray

    from ludwig.data.dataset.ray import RayDataset

    proc_col = "audio_proc"
    feature_name = "audio_0"
    features = {proc_col: {"name": feature_name, "column": feature_name, "type": "audio"}}
    training_set_metadata = {feature_name: _lazy_audio_metadata(feature_dim, max_length)}
    df = pd.DataFrame({proc_col: paths})
    ray_ds = RayDataset.__new__(RayDataset)
    ray_ds.ds = ray.data.from_pandas(df)
    ray_ds.features = features
    ray_ds.training_set_metadata = training_set_metadata
    ray_ds.data_cache_fp = None
    ray_ds.data_parquet_fp = None

    all_timings = []
    for epoch in range(epochs):
        with ray_ds.initialize_batcher(batch_size=batch_size, should_shuffle=False) as batcher:
            step = 0
            while not batcher.last_batch():
                t0 = time.perf_counter()
                _batch = batcher.next_batch()
                t1 = time.perf_counter()
                time.sleep(gpu_work_s)
                t2 = time.perf_counter()
                global_step = epoch * batcher.steps_per_epoch + step
                if global_step >= n_warmup:
                    all_timings.append({"t_fetch": t1 - t0, "t_gpu": t2 - t1, "epoch": epoch, "step": global_step})
                step += 1

    return all_timings


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

_COL = 22


def _fmt_stat(s: dict, unit: str = "ms") -> str:
    return f"{s['mean']:6.1f} (p95={s['p95']:5.1f}) {unit}"


def _print_report(results: dict, n_samples: int, n_epochs: int, gpu_work_ms: float):
    print()
    print(f"{'Mode':<26} {'t_fetch mean(p95)':>22} {'t_gpu mean(p95)':>22} {'util%':>7} {'sps':>8}")
    print("─" * 90)
    for mode, r in results.items():
        if not r:
            print(f"  {mode:<24}  FAILED")
            continue
        fetch = _fmt_stat(r["fetch_ms"])
        gpu = _fmt_stat(r["gpu_ms"])
        util = f"{r['util_pct']['mean']:5.1f}%"
        sps = f"{r['sps']:>8,.0f}"
        print(f"  {mode:<24}  {fetch:>22}  {gpu:>22}  {util}  {sps}")
    print()

    # Show GPU-idle time breakdown
    print("GPU idle analysis (t_fetch / (t_fetch + t_gpu) × 100 = idle %):")
    baseline_sps = None
    for mode, r in results.items():
        if not r:
            continue
        idle = 100 - r["util_pct"]["mean"]
        overhead = ""
        if baseline_sps is None:
            baseline_sps = r["sps"]
        elif baseline_sps and r["sps"]:
            ratio = baseline_sps / r["sps"]
            overhead = f"  ({ratio:.1f}× slower than {list(results.keys())[0]})"
        print(f"  {mode:<26}  GPU idle {idle:5.1f}%{overhead}")
    print()
    print(f"  GPU work per step: {gpu_work_ms:.0f} ms (simulated)")
    print(f"  Decode budget to match eager: ≤ {gpu_work_ms:.0f} ms per batch")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=23)
    parser.add_argument(
        "--gpu-work-ms",
        type=float,
        default=30.0,
        help="Simulated GPU step time per batch in ms. Try 5 (small model) / 30 / 100 (large model).",
    )
    parser.add_argument("--n-warmup", type=int, default=3, help="Steps to discard for warm-up")
    parser.add_argument("--skip-ray", action="store_true", help="Skip the Ray backend benchmarks")
    args = parser.parse_args()

    gpu_work_s = args.gpu_work_ms / 1000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nGenerating {args.n_samples} WAV files ...", flush=True)
        paths = _write_wav_files(tmpdir, args.n_samples)
        print("  done.")

        ray_ready = False
        if not args.skip_ray:
            try:
                import ray

                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)
                ray_ready = True
                print("Ray initialised.")
            except Exception as e:
                print(f"Ray not available ({e}), skipping lazy_ray.")

        kw = {
            "paths": paths,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "feature_dim": args.feature_dim,
            "max_length": args.max_length,
            "gpu_work_s": gpu_work_s,
            "n_warmup": args.n_warmup,
        }

        configs = [
            ("eager_local", lambda: bench_eager_local(**kw)),
            ("lazy_local_sync", lambda: bench_lazy_local(**kw, prefetch_size=0)),
            ("lazy_local_pre2", lambda: bench_lazy_local(**kw, prefetch_size=2)),
            ("lazy_local_pre4", lambda: bench_lazy_local(**kw, prefetch_size=4)),
        ]
        if ray_ready:
            configs.append(("lazy_ray", lambda: bench_lazy_ray(**kw)))

        print(
            f"\n{'─' * 70}\n"
            f"  n_samples={args.n_samples}  batch_size={args.batch_size}  "
            f"epochs={args.epochs}  gpu_work={args.gpu_work_ms:.0f}ms\n"
            f"{'─' * 70}\n"
        )

        results = {}
        for name, fn in configs:
            print(f"Running {name} ...", flush=True)
            try:
                timings = fn()
                results[name] = _analyze(timings, args.n_samples, args.epochs)
                r = results[name]
                print(
                    f"  fetch {r['fetch_ms']['mean']:.1f}ms  gpu {r['gpu_ms']['mean']:.1f}ms  util {r['util_pct']['mean']:.1f}%  {r['sps']:,.0f} sps"
                )
            except Exception as e:
                import traceback

                results[name] = None
                print(f"  FAILED: {e}")
                traceback.print_exc()

        _print_report(results, args.n_samples, args.epochs, args.gpu_work_ms)

        if ray_ready:
            ray.shutdown()


if __name__ == "__main__":
    main()
