#!/usr/bin/env python3
"""Benchmark lazy media decode throughput: local (LazyColumn) vs Ray (map_batches).

Measures batch iteration speed for audio features with lazy=True across three paths:
  1. eager_local   ��� all files decoded upfront into numpy arrays, then iterated (best-case baseline)
  2. lazy_local    — PandasDataset + LazyColumn, decode per-batch via ThreadPoolExecutor
  3. lazy_ray      — RayDataset + _with_lazy_decode, decode per-batch via Ray map_batches

Run:
    python scripts/benchmark_lazy_decode.py [--n-samples N] [--batch-size B] [--epochs E]

Requirements: soundfile, ray[data], torchaudio (already installed in the Ludwig environment).
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _write_wav_files(dest_dir: str, n: int, duration_s: float = 0.5, sample_rate: int = 16_000) -> list[str]:
    """Write N silent WAV files and return their paths."""
    import torch
    import torchaudio

    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    n_samples = int(duration_s * sample_rate)
    silence = torch.zeros(1, n_samples)  # (channels, samples)
    for i in range(n):
        p = os.path.join(dest_dir, f"audio_{i:05d}.wav")
        torchaudio.save(p, silence, sample_rate)
        paths.append(p)
    return paths


def _make_lazy_audio_metadata(feature_dim: int = 8, max_length: int = 23) -> dict:
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


def _make_features(proc_col: str, feature_name: str) -> dict:
    return {proc_col: {"name": feature_name, "column": feature_name, "type": "audio"}}


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def bench_eager_local(paths: list[str], batch_size: int, epochs: int, feature_dim: int, max_length: int) -> float:
    """Pre-decode all files, then iterate batches from a plain numpy array."""
    from ludwig.features.audio_feature import AudioFeatureMixin

    # Build decode function and decode everything upfront
    meta = _make_lazy_audio_metadata(feature_dim, max_length)["lazy_audio_params"]
    decode_fn = AudioFeatureMixin._make_lazy_decode_fn(
        audio_feature_dict=meta["audio_feature_dict"],
        feature_dim=meta["feature_dim"],
        max_length=meta["max_length"],
        padding_value=meta["padding_value"],
        normalization_type=meta["normalization_type"],
    )
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(16, len(paths))) as ex:
        decoded = np.stack(list(ex.map(decode_fn, paths)))  # (N, f, t)

    total_batches = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        for start in range(0, len(decoded), batch_size):
            batch = decoded[start : start + batch_size]
            _ = batch.sum()  # simulate minimal usage
            total_batches += 1
    elapsed = time.perf_counter() - t0
    return len(paths) * epochs / elapsed  # samples/sec


def bench_lazy_local(paths: list[str], batch_size: int, epochs: int, feature_dim: int, max_length: int) -> float:
    """PandasDataset + LazyColumn decode-per-batch."""
    from ludwig.data.dataset.pandas import PandasDataset

    proc_col = "audio_proc"
    feature_name = "audio_0"
    features = _make_features(proc_col, feature_name)
    training_set_metadata = {feature_name: _make_lazy_audio_metadata(feature_dim, max_length)}

    dataset_dict = {proc_col: np.array(paths, dtype=object)}
    ds = PandasDataset(dataset_dict, features, data_cache_fp=None, training_set_metadata=training_set_metadata)

    total_batches = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        with ds.initialize_batcher(batch_size=batch_size, should_shuffle=False) as batcher:
            while not batcher.last_batch():
                batch = batcher.next_batch()
                _ = batch[proc_col].sum()
                total_batches += 1
    elapsed = time.perf_counter() - t0
    return len(paths) * epochs / elapsed


def bench_eager_ray(paths: list[str], batch_size: int, epochs: int, feature_dim: int, max_length: int) -> float:
    """Ray dataset with pre-decoded tensors (lazy=False baseline) — no decode overhead at batch time."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    import ray

    from ludwig.data.dataset.ray import RayDataset
    from ludwig.features.audio_feature import AudioFeatureMixin

    proc_col = "audio_proc"
    feature_name = "audio_0"
    # Decode everything upfront (simulates lazy=False behaviour)
    meta = _make_lazy_audio_metadata(feature_dim, max_length)["lazy_audio_params"]
    decode_fn = AudioFeatureMixin._make_lazy_decode_fn(
        audio_feature_dict=meta["audio_feature_dict"],
        feature_dim=meta["feature_dim"],
        max_length=meta["max_length"],
        padding_value=meta["padding_value"],
        normalization_type=meta["normalization_type"],
    )
    with ThreadPoolExecutor(max_workers=min(16, len(paths))) as ex:
        decoded = list(ex.map(decode_fn, paths))

    features = {proc_col: {"name": feature_name, "column": feature_name, "type": "audio"}}
    # No lazy in metadata → _with_lazy_decode is a no-op
    training_set_metadata = {feature_name: {"lazy": False, "reshape": (feature_dim, max_length)}}

    df = pd.DataFrame({proc_col: decoded})
    ray_ds = RayDataset.__new__(RayDataset)
    ray_ds.ds = ray.data.from_pandas(df)
    ray_ds.features = features
    ray_ds.training_set_metadata = training_set_metadata
    ray_ds.data_cache_fp = None
    ray_ds.data_parquet_fp = None

    total_batches = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        with ray_ds.initialize_batcher(batch_size=batch_size, should_shuffle=False) as batcher:
            while not batcher.last_batch():
                batch = batcher.next_batch()
                _ = batch[proc_col].sum()
                total_batches += 1
    elapsed = time.perf_counter() - t0
    return len(paths) * epochs / elapsed


def bench_lazy_ray(paths: list[str], batch_size: int, epochs: int, feature_dim: int, max_length: int) -> float:
    """RayDataset + _with_lazy_decode decode-per-batch."""
    import pandas as pd
    import ray

    from ludwig.data.dataset.ray import RayDataset

    proc_col = "audio_proc"
    feature_name = "audio_0"
    features = _make_features(proc_col, feature_name)
    training_set_metadata = {feature_name: _make_lazy_audio_metadata(feature_dim, max_length)}

    df = pd.DataFrame({proc_col: paths})
    ray_ds = RayDataset.__new__(RayDataset)
    ray_ds.ds = ray.data.from_pandas(df)
    ray_ds.features = features
    ray_ds.training_set_metadata = training_set_metadata
    ray_ds.data_cache_fp = None
    ray_ds.data_parquet_fp = None

    total_batches = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        with ray_ds.initialize_batcher(batch_size=batch_size, should_shuffle=False) as batcher:
            while not batcher.last_batch():
                batch = batcher.next_batch()
                _ = batch[proc_col].sum()
                total_batches += 1
    elapsed = time.perf_counter() - t0
    return len(paths) * epochs / elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=200, help="Number of audio files to generate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for iteration")
    parser.add_argument("--epochs", type=int, default=3, help="Number of full passes over the dataset")
    parser.add_argument("--feature-dim", type=int, default=8, help="Audio feature dim (num_filter_bands)")
    parser.add_argument("--max-length", type=int, default=23, help="Audio max length (frames)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Generating {args.n_samples} WAV files ...", flush=True)
        paths = _write_wav_files(tmpdir, args.n_samples)
        print("  done.\n")

        print("Initialising Ray ...", flush=True)
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)
        print("  done.\n")

        configs = [
            ("eager_local", bench_eager_local),
            ("lazy_local ", bench_lazy_local),
            ("eager_ray  ", bench_eager_ray),
            ("lazy_ray   ", bench_lazy_ray),
        ]

        results = {}
        for name, fn in configs:
            print(f"Running {name.strip()} ...", flush=True)
            try:
                sps = fn(paths, args.batch_size, args.epochs, args.feature_dim, args.max_length)
                results[name] = sps
                print(f"  {sps:,.0f} samples/sec")
            except Exception as e:
                results[name] = None
                print(f"  FAILED: {e}")

        print("\n--- Summary ---")
        eager_ray_sps = results.get("eager_ray  ")
        for name, sps in results.items():
            if sps is None:
                print(f"  {name}: FAILED")
            else:
                print(f"  {name}: {sps:>10,.0f} sps")

        print()
        if eager_ray_sps and results.get("lazy_ray   "):
            overhead_pct = (eager_ray_sps - results["lazy_ray   "]) / eager_ray_sps * 100
            print(f"  lazy_ray overhead vs eager_ray: {overhead_pct:.0f}%")
            print("  (this is the decode-per-batch cost; scales linearly with file I/O speed)")
        print()
        print("Notes:")
        print("  eager_local: pre-decoded in-memory arrays — pure numpy, no I/O, not a fair comparison")
        print("  lazy_local: PandasDataset + LazyColumn — decode per batch via ThreadPoolExecutor")
        print("  eager_ray:  pre-decoded tensors in Ray object store — Ray overhead without decode")
        print("  lazy_ray:   RayDataset + _with_lazy_decode — decode per batch via map_batches")
        print()
        print("  eager_ray vs lazy_ray shows the cost of decode-per-batch in the Ray pipeline.")
        print("  lazy_local vs lazy_ray shows the extra overhead of Ray vs direct numpy access.")
        print("  In distributed training, each worker decodes in parallel, so effective")
        print("  throughput scales with number of workers.")

        ray.shutdown()


if __name__ == "__main__":
    main()
