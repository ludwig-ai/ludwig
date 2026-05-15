"""Benchmark: lazy vs eager preprocessing for image and audio features.

Each mode (eager, lazy) runs in full isolation — separate CSV copies so
Ludwig cannot reuse any preprocessed cache between modes.

Measures per mode:
  1. Preprocessing-only time + peak heap   (model.preprocess on copy A)
  2. Full-pipeline time + peak heap        (model.train on copy B — independent)
  3. Training throughput derived from (2) - (1)

Run:
    python scripts/benchmark_lazy_preprocessing.py [--n-samples N]
Requires torch, torchaudio (via Ludwig venv)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import tempfile
import time
import tracemalloc
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_png_rgb(path: Path, width: int = 64, height: int = 64, seed: int = 0) -> None:
    import zlib

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    def _chunk(t: bytes, d: bytes) -> bytes:
        crc = zlib.crc32(t + d) & 0xFFFFFFFF
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


def _make_wav_mono(path: Path, duration_s: float = 2.0, sample_rate: int = 16000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    freq = 200.0 + rng.integers(0, 600)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_image_dataset(root: Path, n: int, tag: str = "") -> str:
    """Return CSV path; creates N 64×64 PNG images under root/images_{tag}/."""
    img_dir = root / f"images_{tag}"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels = ["cat", "dog", "bird", "fish"]
    rows = []
    for i in range(n):
        p = img_dir / f"img_{i:05d}.png"
        _make_png_rgb(p, seed=i)
        rows.append({"image_path": str(p), "label": labels[i % len(labels)]})
    csv_path = str(root / f"images_{tag}.csv")
    _write_csv(csv_path, rows, ["image_path", "label"])
    return csv_path


def create_audio_dataset(root: Path, n: int, tag: str = "", duration_s: float = 2.0) -> str:
    """Return CSV path; creates N WAV files under root/audio_{tag}/."""
    audio_dir = root / f"audio_{tag}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    labels = ["yes", "no", "up", "down"]
    rows = []
    for i in range(n):
        p = audio_dir / f"clip_{i:05d}.wav"
        _make_wav_mono(p, duration_s=duration_s, seed=i)
        rows.append({"audio_path": str(p), "label": labels[i % len(labels)]})
    csv_path = str(root / f"audio_{tag}.csv")
    _write_csv(csv_path, rows, ["audio_path", "label"])
    return csv_path


# ---------------------------------------------------------------------------
# Memory + timing helpers
# ---------------------------------------------------------------------------


def _tracked(fn):
    """Run fn(); return (result, peak_heap_mb, elapsed_s)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.clear_traces()
    return result, peak / 1024 / 1024, elapsed


# ---------------------------------------------------------------------------
# Ludwig wrappers
# ---------------------------------------------------------------------------


def _image_config(lazy: bool, batch_size: int) -> dict:
    return {
        "input_features": [
            {
                "name": "image_path",
                "type": "image",
                "preprocessing": {"lazy": lazy, "num_channels": 3, "height": 64, "width": 64},
                "encoder": {
                    "type": "stacked_cnn",
                    "conv_layers": [{"num_filters": 8, "filter_size": 3}],
                    "fc_layers": [{"output_size": 16}],
                },
            }
        ],
        "output_features": [{"name": "label", "type": "category"}],
        "trainer": {"epochs": 1, "batch_size": batch_size},
        "combiner": {"type": "concat", "fc_layers": [{"output_size": 16}]},
    }


def _audio_config(lazy: bool, batch_size: int) -> dict:
    return {
        "input_features": [
            {
                "name": "audio_path",
                "type": "audio",
                "preprocessing": {
                    "lazy": lazy,
                    "audio_feature": {"type": "fbank", "num_filter_bands": 80},
                    "audio_file_length_limit_in_s": 2.0,
                    "norm": None,
                },
                "encoder": {
                    "type": "stacked_cnn",
                    "conv_layers": [{"num_filters": 8, "filter_size": 3}],
                    "fc_layers": [{"output_size": 16}],
                },
            }
        ],
        "output_features": [{"name": "label", "type": "category"}],
        "trainer": {"epochs": 1, "batch_size": batch_size},
        "combiner": {"type": "concat", "fc_layers": [{"output_size": 16}]},
    }


def _run_preprocess(csv_path: str, config: dict, out_dir: str) -> None:
    import ludwig.api

    model = ludwig.api.LudwigModel(config=config, logging_level=40)
    model.preprocess(dataset=csv_path, output_directory=out_dir, skip_save_processed_input=False)


def _run_train(csv_path: str, config: dict, out_dir: str) -> None:
    import ludwig.api

    model = ludwig.api.LudwigModel(config=config, logging_level=40)
    model.train(
        dataset=csv_path,
        output_directory=out_dir,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=False,
    )


# ---------------------------------------------------------------------------
# Per-modality benchmark
# ---------------------------------------------------------------------------


def benchmark_modality(modality: str, n_samples: int, batch_size: int, tmpdir: Path) -> dict:
    print(f"\n{'=' * 64}")
    print(f"  {modality.upper()}  ·  {n_samples} samples  ·  batch_size={batch_size}")
    print("=" * 64)

    cfg_fn = _image_config if modality == "image" else _audio_config

    t0 = time.perf_counter()
    # Create THREE independent copies so there is zero shared cache between runs
    if modality == "image":
        csv_prep_eager = create_image_dataset(tmpdir / modality, n_samples, tag="prep_eager")
        csv_prep_lazy = create_image_dataset(tmpdir / modality, n_samples, tag="prep_lazy")
        csv_full_eager = create_image_dataset(tmpdir / modality, n_samples, tag="full_eager")
        csv_full_lazy = create_image_dataset(tmpdir / modality, n_samples, tag="full_lazy")
    else:
        csv_prep_eager = create_audio_dataset(tmpdir / modality, n_samples, tag="prep_eager")
        csv_prep_lazy = create_audio_dataset(tmpdir / modality, n_samples, tag="prep_lazy")
        csv_full_eager = create_audio_dataset(tmpdir / modality, n_samples, tag="full_eager")
        csv_full_lazy = create_audio_dataset(tmpdir / modality, n_samples, tag="full_lazy")
    print(f"  Dataset copies generated in {time.perf_counter() - t0:.1f}s")

    results: dict = {}

    for lazy in [False, True]:
        mode = "lazy" if lazy else "eager"
        config = cfg_fn(lazy=lazy, batch_size=batch_size)
        csv_prep = csv_prep_lazy if lazy else csv_prep_eager
        csv_full = csv_full_lazy if lazy else csv_full_eager

        # ── Preprocessing only ────────────────────────────────────────
        prep_out = str(tmpdir / f"{modality}_{mode}_prep")
        _, prep_heap_mb, prep_s = _tracked(lambda _c=csv_prep, _cfg=config, _o=prep_out: _run_preprocess(_c, _cfg, _o))
        if os.path.exists(prep_out):
            shutil.rmtree(prep_out)

        # ── Full pipeline (preprocess + train) — completely isolated ──
        full_out = str(tmpdir / f"{modality}_{mode}_full")
        _, full_heap_mb, full_s = _tracked(lambda _c=csv_full, _cfg=config, _o=full_out: _run_train(_c, _cfg, _o))
        if os.path.exists(full_out):
            shutil.rmtree(full_out)

        # Training-only time = full pipeline − preprocessing-only
        train_s = max(0.0, full_s - prep_s)
        throughput = n_samples / train_s if train_s > 0 else float("inf")

        print(f"\n  [{mode.upper()}]")
        print(f"    Preprocessing : {prep_s:7.2f}s  peak heap: {prep_heap_mb:7.1f} MB")
        print(f"    Full pipeline : {full_s:7.2f}s  peak heap: {full_heap_mb:7.1f} MB")
        print(f"    Training est. : {train_s:7.2f}s  throughput: {throughput:.1f} samples/s")

        results[mode] = {
            "preprocessing_s": round(prep_s, 3),
            "preprocessing_peak_heap_mb": round(prep_heap_mb, 1),
            "full_pipeline_s": round(full_s, 3),
            "full_pipeline_peak_heap_mb": round(full_heap_mb, 1),
            "training_s_est": round(train_s, 3),
            "training_throughput_samples_per_s": round(throughput, 1) if throughput != float("inf") else None,
        }

    # ── Summary ───────────────────────────────────────────────────────
    e, la = results["eager"], results["lazy"]

    def _ratio(a, b, key):
        return round(a[key] / b[key], 2) if b.get(key) and b[key] > 0 else None

    prep_speedup = _ratio(e, la, "preprocessing_s")
    prep_mem_ratio = _ratio(e, la, "preprocessing_peak_heap_mb")
    full_speedup = _ratio(e, la, "full_pipeline_s")

    lazy_tp = la.get("training_throughput_samples_per_s") or 0
    eager_tp = e.get("training_throughput_samples_per_s") or 0
    tp_ratio = round(lazy_tp / eager_tp, 2) if eager_tp > 0 else None

    print(f"\n  SUMMARY ({modality}):")
    print(f"    Preprocessing speedup (lazy vs eager): {prep_speedup}x faster")
    print(f"    Preprocessing heap ratio (eager/lazy): {prep_mem_ratio}x less heap with lazy")
    print(f"    Full-pipeline speedup  (eager/lazy):  {full_speedup}x")
    print(f"    Training throughput ratio (lazy/eager): {tp_ratio}x")

    results["_summary"] = {
        "preprocessing_speedup_lazy_over_eager": prep_speedup,
        "preprocessing_heap_reduction_eager_over_lazy": prep_mem_ratio,
        "full_pipeline_speedup_eager_over_lazy": full_speedup,
        "training_throughput_ratio_lazy_over_eager": tp_ratio,
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="scripts/benchmark_results.json")
    parser.add_argument("--modalities", nargs="+", default=["image", "audio"], choices=["image", "audio"])
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not available — run from the Ludwig venv")
        return

    print("Ludwig lazy-preprocessing benchmark")
    print(f"  n_samples={args.n_samples}  batch_size={args.batch_size}  modalities={args.modalities}")

    results: dict = {"config": vars(args)}
    with tempfile.TemporaryDirectory(prefix="ludwig_lazy_bench_") as td:
        tmpdir = Path(td)
        for modality in args.modalities:
            try:
                results[modality] = benchmark_modality(modality, args.n_samples, args.batch_size, tmpdir)
            except Exception as exc:
                import traceback

                print(f"\nERROR in {modality}: {exc}")
                traceback.print_exc()
                results[modality] = {"error": str(exc)}

    print(f"\n{'=' * 64}")
    print("Full results:")
    print(json.dumps(results, indent=2))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
