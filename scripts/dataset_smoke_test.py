#!/usr/bin/env python3
"""1-epoch Ludwig smoke test for every dataset config.

Streams 1000 rows directly from HF (no full download), builds a minimal
Ludwig config from the YAML, runs 1 epoch, then wipes the HF cache entry
before moving to the next dataset.

Usage:
    python scripts/dataset_smoke_test.py [--names ds1 ds2 ...] [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import traceback
from collections import Counter
from typing import Any

import datasets as hf_datasets
import pandas as pd
import yaml

logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("LUDWIG_DISABLE_PROGRESS_BAR", "1")

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "ludwig", "datasets", "configs")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "smoke_results.json")
HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
HF_DS_CACHE = os.path.expanduser("~/.cache/huggingface/datasets")

SAMPLE_ROWS = 1000
MIN_ROWS = 32

# Skip: gated/no supervised task/already covered
SKIP = {"imagenet1k", "gigaspeech", "hugging_face"}

ENCODER_OVERRIDES: dict[str, dict] = {
    "text": {"encoder": {"type": "embed", "embedding_size": 16, "trainable": True}},
    "image": {"encoder": {"type": "stacked_cnn", "num_conv_layers": 1, "num_filters": 8, "output_size": 16}},
    "audio": {"encoder": {"type": "stacked_cnn", "num_conv_layers": 1, "num_filters": 8, "output_size": 16}},
    "sequence": {"encoder": {"type": "embed", "embedding_size": 16}},
    "set": {"encoder": {"type": "embed", "embedding_size": 16}},
}


def load_dataset_config(name: str) -> dict:
    path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def stream_sample(
    hf_id: str, hf_sub: str | None, n: int = SAMPLE_ROWS, shuffle_buffer: int = 100000, skip: int = 0
) -> pd.DataFrame | None:
    """Stream up to n rows from HF without downloading the full dataset."""
    try:
        ds_stream = hf_datasets.load_dataset(
            path=hf_id,
            name=hf_sub,
            trust_remote_code=False,
            streaming=True,
        )
        split_name = "train" if "train" in ds_stream else list(ds_stream.keys())[0]
        ds = ds_stream[split_name]
        if skip:
            ds = ds.skip(skip)
        # Shuffle to ensure label diversity in sorted datasets (e.g. dbpedia, imdb).
        # Use a smaller buffer for media datasets to avoid streaming 100k large files.
        ds = ds.shuffle(seed=42, buffer_size=min(n * 100, shuffle_buffer))
        rows = list(ds.take(n))
        if len(rows) < MIN_ROWS:
            return None
        return pd.DataFrame(rows)
    except Exception as e:
        raise RuntimeError(f"Stream failed: {e}") from e


def wipe_hf_cache_for(hf_id: str, hf_sub: str | None) -> None:
    """Delete hub and dataset cache entries for a specific dataset."""
    # Hub cache dirs are named datasets--org--repo
    normalized = hf_id.replace("/", "--")
    for cache_root in [HF_HUB_CACHE, HF_DS_CACHE]:
        if not os.path.isdir(cache_root):
            continue
        for entry in os.listdir(cache_root):
            if normalized in entry or (hf_sub and hf_sub in entry):
                path = os.path.join(cache_root, entry)
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass


def build_ludwig_config(cfg: dict) -> dict:
    out_names = {f["name"] for f in cfg.get("output_features", [])}
    input_features = []
    for col in cfg.get("columns", []):
        if col["name"] in out_names:
            continue
        feat = {"name": col["name"], "type": col["type"]}
        feat.update(ENCODER_OVERRIDES.get(col["type"], {}))
        input_features.append(feat)
    output_features = [{"name": f["name"], "type": f["type"]} for f in cfg.get("output_features", [])]
    return {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "fc_size": 32},
        "trainer": {"epochs": 1, "batch_size": 32, "learning_rate": 0.001, "eval_batch_size": 32},
        "preprocessing": {"split": {"type": "random", "probabilities": [0.7, 0.1, 0.2]}},
    }


def apply_custom_loader(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run the custom loader's _transform on the raw dataframe."""
    cfg = load_dataset_config(name)
    loader_spec = cfg.get("loader", "")
    if not loader_spec or loader_spec == "hugging_face.HFLoader":
        return df

    module_name, cls_name = loader_spec.rsplit(".", 1)
    full_module = f"ludwig.datasets.loaders.{module_name}"
    try:
        import importlib

        mod = importlib.import_module(full_module)
        cls = getattr(mod, cls_name)

        # processed_dataset_dir is a read-only @property — override via subclass
        _tmpdir = tempfile.mkdtemp(prefix=f"ludwig_smoke_{name}_")
        PatchedCls = type(
            f"_{cls_name}",
            (cls,),
            {
                "processed_dataset_dir": property(lambda self, d=_tmpdir: d),
            },
        )
        instance = object.__new__(PatchedCls)
        return instance._transform(df)
    except Exception as e:
        raise RuntimeError(f"Loader {loader_spec} failed: {e}") from e


def _materialize_media_columns(df: pd.DataFrame, tmpdir: str) -> pd.DataFrame:
    """Replace PIL images and HF audio objects with file paths."""
    import io

    try:
        from PIL import Image as PILImage

        _pil_available = True
    except ImportError:
        _pil_available = False

    df = df.copy()
    for col in df.columns:
        sample = df[col].dropna().iloc[:1]
        if sample.empty:
            continue
        val = sample.iloc[0]

        # PIL image or HF image dict
        is_image = _pil_available and (
            isinstance(val, PILImage.Image)
            or (isinstance(val, dict) and ("bytes" in val or "path" in val) and "array" not in val)
        )
        # HF audio dict {'array': ..., 'sampling_rate': ...} or TorchCodec object
        is_audio = (isinstance(val, dict) and "array" in val and "sampling_rate" in val) or (
            hasattr(val, "__class__") and "AudioDecoder" in type(val).__name__
        )

        if is_image:
            img_dir = os.path.join(tmpdir, col)
            os.makedirs(img_dir, exist_ok=True)
            paths = []
            for idx, v in df[col].items():
                path = os.path.join(img_dir, f"{idx}.jpg")
                try:
                    if isinstance(v, PILImage.Image):
                        v.convert("RGB").save(path, format="JPEG")
                    elif isinstance(v, dict) and "bytes" in v and v["bytes"]:
                        PILImage.open(io.BytesIO(v["bytes"])).convert("RGB").save(path, format="JPEG")
                    else:
                        path = ""
                except Exception:
                    path = ""
                paths.append(path)
            df[col] = paths

        elif is_audio:
            import numpy as np

            try:
                import soundfile as sf

                _sf_available = True
            except ImportError:
                _sf_available = False
            aud_dir = os.path.join(tmpdir, col)
            os.makedirs(aud_dir, exist_ok=True)
            paths = []
            for idx, v in df[col].items():
                path = os.path.join(aud_dir, f"{idx}.wav")
                try:
                    if isinstance(v, dict) and "array" in v:
                        arr = np.array(v["array"], dtype=np.float32)
                        sr = int(v.get("sampling_rate", 16000))
                        if _sf_available:
                            sf.write(path, arr, sr)
                        else:
                            from scipy.io import wavfile

                            wavfile.write(path, sr, (arr * 32767).astype(np.int16))
                    elif hasattr(v, "get_all_samples"):
                        samples = v.get_all_samples()
                        arr = samples.data.numpy()
                        sr = int(samples.sample_rate)
                        if arr.ndim > 1:
                            arr = arr[0]  # first channel
                        arr = arr.astype(np.float32)
                        if _sf_available:
                            sf.write(path, arr, sr)
                        else:
                            from scipy.io import wavfile

                            wavfile.write(path, sr, (arr * 32767).astype(np.int16))
                    else:
                        path = ""
                except Exception:
                    path = ""
                paths.append(path)
            df[col] = paths

    return df


def run_smoke_test(name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "status": "unknown", "error": None, "rows": 0}

    if name in SKIP:
        result["status"] = "skipped"
        return result

    try:
        cfg = load_dataset_config(name)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"YAML load failed: {e}"
        return result

    hf_id = cfg.get("huggingface_dataset_id")
    hf_sub = cfg.get("huggingface_subsample")

    if not hf_id:
        result["status"] = "skipped"
        result["error"] = "Not an HF dataset (pre-existing Ludwig dataset)"
        return result

    # 1. Stream 1000 rows — use small shuffle buffer for media datasets to avoid
    #    streaming 100k large files; text datasets can afford the large buffer
    #    to ensure label diversity in sorted datasets (e.g. dbpedia_14, imdb).
    col_types = {col["type"] for col in cfg.get("columns", [])}
    has_media = bool(col_types & {"audio", "image"})
    shuffle_buf = 5000 if has_media else 100000
    try:
        df = stream_sample(hf_id, hf_sub, SAMPLE_ROWS, shuffle_buffer=shuffle_buf)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]
        return result

    if df is None:
        result["status"] = "error"
        result["error"] = f"Too few rows (< {MIN_ROWS})"
        return result

    result["rows"] = len(df)

    # 2. Apply custom loader transform if needed
    try:
        df = apply_custom_loader(name, df)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Custom loader failed: {e}"
        return result

    # 3. Materialize any PIL images to disk before passing to Ludwig
    try:
        _img_tmp = tempfile.mkdtemp(prefix=f"ludwig_imgs_{name}_")
        df = _materialize_media_columns(df, _img_tmp)
    except ImportError:
        pass  # PIL not available — skip

    # 4. Keep only columns declared in the config; drop extras
    cfg_cols = {f["name"] for f in cfg.get("columns", [])}
    df = df[[c for c in df.columns if c in cfg_cols or c == "split"]]

    # 5. Verify output columns exist and have values; retry with skip for sorted datasets
    out_names = [f["name"] for f in cfg.get("output_features", [])]
    out_types = {f["name"]: f["type"] for f in cfg.get("output_features", [])}
    for oc in out_names:
        if oc not in df.columns:
            result["status"] = "error"
            result["error"] = f"Output column '{oc}' missing after transform. Cols: {list(df.columns)}"
            return result
        if df[oc].isna().all():
            result["status"] = "error"
            result["error"] = f"Output column '{oc}' is all-null"
            return result
        # For category/binary outputs: if only 1 distinct non-null value, the dataset
        # is likely sorted. Retry by skipping 40k rows to sample from a different region.
        if out_types.get(oc) in ("category", "binary") and df[oc].dropna().nunique() < 2:
            try:
                df2 = stream_sample(hf_id, hf_sub, SAMPLE_ROWS // 2, shuffle_buffer=shuffle_buf, skip=40000)
                if df2 is not None:
                    df2 = apply_custom_loader(name, df2)
                    df2 = _materialize_media_columns(df2, _img_tmp)
                    df2 = df2[[c for c in df2.columns if c in cfg_cols or c == "split"]]
                    df = pd.concat([df[: SAMPLE_ROWS // 2], df2], ignore_index=True)
            except Exception:
                pass  # keep original df

    # 6. Build Ludwig config and train
    ludwig_cfg = build_ludwig_config(cfg)
    if not ludwig_cfg["input_features"]:
        result["status"] = "error"
        result["error"] = "No input features"
        return result

    try:
        from ludwig.api import LudwigModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(ludwig_cfg, logging_level=logging.ERROR)
            model.train(
                dataset=df,
                output_directory=tmpdir,
                skip_save_training_description=True,
                skip_save_training_statistics=True,
                skip_save_model=True,
                skip_save_progress=True,
                skip_save_log=True,
                skip_save_processed_input=True,
            )
        result["status"] = "pass"
    except Exception as e:
        result["status"] = "fail"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["traceback"] = traceback.format_exc()[-600:]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    all_names = sorted(
        f.replace(".yaml", "") for f in os.listdir(CONFIGS_DIR) if f.endswith(".yaml") and not f.startswith("__")
    )
    names = [n for n in args.names if n in set(all_names)] if args.names else all_names

    existing: dict[str, dict] = {}
    if args.resume and os.path.exists(RESULTS_PATH):
        try:
            for r in json.load(open(RESULTS_PATH)):
                existing[r["name"]] = r
        except Exception:
            pass

    results: list[dict] = []
    total = len(names)
    for i, name in enumerate(names):
        if args.resume and name in existing and existing[name]["status"] == "pass":
            results.append(existing[name])
            print(f"[{i + 1}/{total}] SKIP {name} (already passed)")
            continue

        cfg = {}
        try:
            cfg = load_dataset_config(name)
        except Exception:
            pass
        hf_id = cfg.get("huggingface_dataset_id", "")
        hf_sub = cfg.get("huggingface_subsample")

        print(f"[{i + 1}/{total}] {name}...", end=" ", flush=True)
        r = run_smoke_test(name)
        results.append(r)

        sym = {"pass": "✓", "fail": "✗", "error": "E", "skipped": "—"}.get(r["status"], "?")
        print(f"{sym} [{r['status']}]")
        if r.get("error"):
            print(f"    {r['error'][:120]}")

        # Wipe HF cache for this dataset immediately
        if hf_id:
            wipe_hf_cache_for(hf_id, hf_sub)

        # Save results after every dataset
        all_results = {**existing, **{rr["name"]: rr for rr in results}}
        with open(RESULTS_PATH, "w") as f:
            json.dump(list(all_results.values()), f, indent=2)

    status_counts = Counter(r["status"] for r in results)
    print("\n=== Smoke Test Summary ===")
    for s, n in status_counts.most_common():
        print(f"  {s}: {n}")
    print(f"  Total: {len(results)}")

    print("\nFAILED:")
    for r in results:
        if r["status"] == "fail":
            print(f"  {r['name']}: {(r.get('error') or '')[:100]}")

    print("\nERRORS:")
    for r in results:
        if r["status"] == "error":
            print(f"  {r['name']}: {(r.get('error') or '')[:100]}")


if __name__ == "__main__":
    main()
