#!/usr/bin/env python3
"""Probe HuggingFace datasets, auto-generate YAML configs, and produce a smoke-test report.

Usage:
    python scripts/probe_hf_datasets.py [--start N] [--end N] [--resume]

The script:
1. Streams 20 rows from each candidate dataset (no full download)
2. Inspects column types
3. Auto-generates a YAML config for simple datasets
4. Writes results to scripts/probe_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from collections import Counter
from typing import Any

import datasets as hf_datasets
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "ludwig", "datasets", "configs")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "probe_results.json")
CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "hf_dataset_candidates.json")

EXISTING = {p.replace(".yaml", "") for p in os.listdir(CONFIGS_DIR) if p.endswith(".yaml")}

LUDWIG_TYPES = {"text", "category", "binary", "number", "image", "audio", "sequence", "set", "vector", "bag"}

_OUTPUT_COL_CANDIDATES = [
    "label",
    "labels",
    "target",
    "output",
    "class",
    "category",
    "sentiment",
    "answer",
    "score",
    "rating",
    "tag",
    "tags",
    "intent",
    "emotion",
    "polarity",
    "stance",
    "result",
    "verdict",
    "quality",
]


# ── Column-type inference ────────────────────────────────────────────────────


def infer_ludwig_type(col: pd.Series, feature_info=None) -> str:
    dtype = col.dtype

    if feature_info is not None:
        fname = str(type(feature_info).__name__)
        if "ClassLabel" in fname:
            return "category"
        if "Sequence" in fname:
            return "_list"
        if "Image" in fname:
            return "image"
        if "Audio" in fname:
            return "audio"
        if "Translation" in fname:
            return "_translation"

    if dtype is object:
        sample = col.dropna().head(20)
        if len(sample) == 0:
            return "text"
        first = sample.iloc[0]
        if isinstance(first, (list, tuple)):
            return "_list"
        if isinstance(first, dict):
            return "_dict"
        if isinstance(first, (bytes, bytearray)):
            return "_bytes"
        unique_ratio = col.nunique() / max(len(col), 1)
        avg_len = sample.apply(lambda x: len(str(x))).mean()
        if unique_ratio < 0.05 and avg_len < 50:
            return "category"
        return "text"

    if pd.api.types.is_bool_dtype(dtype):
        return "binary"
    if pd.api.types.is_integer_dtype(dtype):
        return "category" if col.nunique() <= 20 else "number"
    if pd.api.types.is_float_dtype(dtype):
        return "number"
    return "text"


def infer_columns(df: pd.DataFrame, hf_features) -> dict[str, str]:
    result = {}
    for col in df.columns:
        if col == "split":
            continue
        fi = hf_features.get(col) if hf_features else None
        result[col] = infer_ludwig_type(df[col], fi)
    return result


# ── Output column resolution ─────────────────────────────────────────────────


def _resolve_output_cols(entry: dict, columns: dict[str, str]) -> list[str] | None:
    """Handle both column-name and type-name forms for output_features."""
    raw = entry.get("output_features", [])
    if not raw:
        return None

    col_names = set(columns.keys())
    # If all values are actual column names, use them directly
    if all(oc in col_names for oc in raw):
        return raw

    # Agent returned types ("category", "binary") not column names — auto-detect
    for cand in _OUTPUT_COL_CANDIDATES:
        if cand in col_names and not columns[cand].startswith("_"):
            return [cand]

    # Fallback: first column with category/binary/number type that isn't an id
    for col, typ in columns.items():
        if col in ("idx", "id", "index") or typ.startswith("_"):
            continue
        if typ in ("category", "binary", "number"):
            return [col]
    return None


# ── YAML generation ──────────────────────────────────────────────────────────

YAML_TMPL = """\
version: 1.0
name: {name}
huggingface_dataset_id: {hf_id}
{subsample_line}loader: hugging_face.HFLoader
description: |
  {description}
columns:
{columns_block}
output_features:
{output_block}
"""


def make_yaml(entry: dict, columns: dict[str, str]) -> tuple[str | None, list[str] | None]:
    """Return (yaml_string, output_cols) or (None, None)."""
    for t in columns.values():
        if t.startswith("_"):
            return None, None

    output_cols = _resolve_output_cols(entry, columns)
    if not output_cols:
        return None, None
    for oc in output_cols:
        if oc not in columns:
            return None, None

    subsample = entry.get("hf_subsample")
    subsample_line = f"huggingface_subsample: {subsample}\n" if subsample else ""
    columns_block = "\n".join(
        f"  - name: {col}\n    type: {typ}" for col, typ in columns.items() if not col.startswith("_")
    )
    output_block = "\n".join(f"  - name: {oc}\n    type: {columns[oc]}" for oc in output_cols if oc in columns)
    notes = entry.get("notes", "") or entry.get("name", "")
    yaml_str = YAML_TMPL.format(
        name=entry["name"],
        hf_id=entry["hf_id"],
        subsample_line=subsample_line,
        description=notes,
        columns_block=columns_block,
        output_block=output_block,
    )
    return yaml_str, output_cols


# ── Dataset probing ──────────────────────────────────────────────────────────


def probe_one(entry: dict) -> dict[str, Any]:
    hf_id = entry["hf_id"]
    hf_sub = entry.get("hf_subsample")
    name = entry["name"]

    result: dict[str, Any] = {
        "name": name,
        "hf_id": hf_id,
        "hf_subsample": hf_sub,
        "task": entry.get("task", ""),
        "status": "unknown",
        "columns": {},
        "rows": -1,
        "splits": [],
        "yaml_written": False,
        "output_cols": [],
        "error": None,
        "needs_custom_loader": entry.get("needs_custom_loader", False),
    }

    if name in EXISTING:
        result["status"] = "already_exists"
        return result

    try:
        ds_stream = hf_datasets.load_dataset(
            path=hf_id,
            name=hf_sub,
            trust_remote_code=False,
            streaming=True,
        )
        split_name = "train" if "train" in ds_stream else list(ds_stream.keys())[0]
        ds = ds_stream[split_name]
        rows = list(ds.take(20))
        if not rows:
            result["status"] = "error"
            result["error"] = "No rows returned"
            return result

        df = pd.DataFrame(rows)
        hf_features = ds.features if hasattr(ds, "features") else None
        columns = infer_columns(df, hf_features)
        result["columns"] = columns
        result["splits"] = list(ds_stream.keys())

        has_complex = any(t.startswith("_") for t in columns.values())
        result["needs_custom_loader"] = has_complex or entry.get("needs_custom_loader", False)

        yaml_str, output_cols = make_yaml(entry, columns)
        if yaml_str and not result["needs_custom_loader"]:
            out_path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
            with open(out_path, "w") as f:
                f.write(yaml_str)
            result["yaml_written"] = True
            result["output_cols"] = output_cols
            result["status"] = "auto_generated"
        elif result["needs_custom_loader"]:
            result["status"] = "needs_custom_loader"
        else:
            result["status"] = "skipped_no_yaml"
            # Store why: which output cols were tried
            result["debug_output_cols"] = _resolve_output_cols(entry, columns)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        logger.debug(traceback.format_exc())

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)
    candidates = candidates[args.start : args.end]

    existing_results: dict[str, dict] = {}
    if args.resume and os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                for r in json.load(f):
                    existing_results[r["name"]] = r
        except Exception:
            pass

    results = []
    total = len(candidates)
    for i, entry in enumerate(candidates):
        name = entry["name"]
        if args.resume and name in existing_results:
            results.append(existing_results[name])
            print(f"[{i + 1}/{total}] SKIP (resumed) {name}")
            continue

        print(f"[{i + 1}/{total}] Probing {name} ({entry['hf_id']})...", end=" ", flush=True)
        r = probe_one(entry)
        results.append(r)

        sym = {
            "auto_generated": "✓",
            "already_exists": "=",
            "needs_custom_loader": "~",
            "error": "✗",
            "skipped_no_yaml": "?",
        }.get(r["status"], "?")
        print(f"{sym} [{r['status']}]")
        if r["error"]:
            print(f"    ERROR: {r['error'][:100]}")

        # Merge with any resumed results and save
        all_results = {**existing_results, **{rr["name"]: rr for rr in results}}
        with open(RESULTS_PATH, "w") as f:
            json.dump(list(all_results.values()), f, indent=2)

    status_counts = Counter(r["status"] for r in results)
    print("\n=== Summary ===")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")
    print(f"  Total: {len(results)}")

    print("\nAuto-generated:")
    for r in results:
        if r["status"] == "auto_generated":
            print(f"  {r['name']} → output: {r.get('output_cols')}")

    print("\nNeeds custom loader:")
    for r in results:
        if r["status"] == "needs_custom_loader":
            complex_cols = [c for c, t in r["columns"].items() if t.startswith("_")]
            print(f"  {r['name']} ({r['task']}): complex={complex_cols}")

    print("\nErrors:")
    for r in results:
        if r["status"] == "error":
            print(f"  {r['name']}: {r['error'][:80]}")


if __name__ == "__main__":
    main()
