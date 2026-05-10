# Ludwig Mega-AutoML Benchmark Plan

> **Goal:** Run the largest machine learning / AutoML experiment ever attempted — hundreds of tabular datasets × ~100 Ludwig configs each — to produce a definitive, reproducible benchmark of deep learning AutoML at scale.

______________________________________________________________________

## 1. Context and Ambition

### Where We Stand vs. the Literature

The largest AutoML benchmarks to date:

| Benchmark    | Year | Total Evaluations    | Datasets                   | Configs/Frameworks               |
| ------------ | ---- | -------------------- | -------------------------- | -------------------------------- |
| **TabArena** | 2025 | ~25 million          | 51 (from 1,053 candidates) | 16 models × up to 200 HP configs |
| **TabRepo**  | 2024 | ~786,000 predictions | 200 (OpenML)               | 1,310 configs, 6 AutoML systems  |
| **TabZilla** | 2023 | ~538,650 models      | 176 (OpenML)               | 19 algorithms × 30 HP configs    |
| **AMLB**     | 2024 | ~40,000 CPU-hrs/fw   | 104                        | 11 AutoML frameworks             |

**No benchmark has ever run Ludwig at scale**, despite Ludwig having 53 encoders, 16 combiners, 13 decoders, and 14 feature types — a combinatorial search space orders of magnitude larger than what any existing benchmark has explored for deep learning models on tabular data.

**Our target:** 300–500 datasets × 100 configs each = **30,000–50,000 model training runs**, exceeding TabZilla by 10×–100× in total deep-learning-specific evaluations.

### What Makes This Unprecedented

1. **Depth of DL architecture search** — TabArena uses 16 models total; we explore ~100 Ludwig-specific DL architectures per dataset
1. **Mixed modality coverage** — Ludwig handles text + tabular + image together; we'll include multimodal Kaggle datasets that no prior benchmark has covered at this scale
1. **Config diversity** — we systematically sample across all combiners (TabNet, Transformer, TabTransformer, FT-Transformer, Concat, etc.), not just tabular GBDTs
1. **Public, reproducible** — every config, every result, every dataset split, fully versioned and OSS

______________________________________________________________________

## 2. Current State Assessment

### 2.1 Ludwig Datasets Infrastructure — What We Have

- **93 built-in datasets** (39 tabular, 45 NLP, 5 vision, 4 multimodal)
- **17 Kaggle competition datasets + 17 Kaggle dataset IDs** hardcoded
- **Download/extract/cache pipeline** (`DatasetLoader` base class, Parquet output)
- **Kaggle API wrapper** (`ludwig/datasets/kaggle.py`) — auth via env vars or `~/.kaggle/kaggle.json`
- **HuggingFace loader** (`hf://` prefix, any HF dataset on demand)
- **14 format readers** (CSV, Parquet, JSON, JSONL, Excel, HDF5, ORC, Feather, etc.)
- **SHA256 checksum validation** per downloaded file
- **16 pre-built model configs** for 15 datasets (default + best quality)

### 2.2 Ludwig AutoML Capabilities — What We Have

- **`auto_train()`** — time-budget-constrained hyperopt with auto config generation
- **13 search algorithms** (random, BOHB, Optuna, Ax, HEBO, SkOpt, Nevergrad, etc.)
- **Ray Tune executor** — multi-node, ASHA scheduling, fractional GPU support
- **Native Optuna executor** (v0.15+) — single machine, SQLite/PostgreSQL resumability
- **53 registered encoders, 13 decoders, 16 combiners**
- **Feature type inference** (rule-based: 15 types including multimodal)
- **Memory-aware tuning** (`memory_tune_config()`)
- **Default search spaces**: TabNet (11 params), Concat (5 params), BERT (4 tuned)
- **Results**: `HyperoptResults`, `hyperopt_stats.json`, Ray Tune `ExperimentAnalysis`

### 2.3 Gaps — What We're Missing

#### Dataset Pipeline Gaps

| Gap                                                                                 | Impact                                        | Priority     |
| ----------------------------------------------------------------------------------- | --------------------------------------------- | ------------ |
| No OpenML integration (5,300+ curated datasets with defined tasks)                  | Can't use the most benchmarked dataset corpus | **Critical** |
| No automated Kaggle dataset scraping/curation                                       | Only 34 Kaggle datasets hardcoded             | **High**     |
| No auto target column detection from raw CSV                                        | Must specify target manually                  | **High**     |
| No dataset quality filtering (leakage detection, constant columns, near-duplicates) | Bad datasets corrupt benchmark                | **High**     |
| No feature type compatibility checking for a given (dataset, config) pair           | Configs fail at runtime                       | **High**     |
| No multi-dataset parallelism (Ludwig schedules trials within one dataset)           | Can't run N datasets × M configs efficiently  | **Critical** |
| No dataset metadata store (n_rows, n_cols, feature types, class imbalance)          | Can't filter or stratify at analysis time     | **Medium**   |

#### Config Generation Gaps

| Gap                                                                                | Impact                                      | Priority     |
| ---------------------------------------------------------------------------------- | ------------------------------------------- | ------------ |
| AutoML generates ONE config; no systematic enumeration of 100 diverse configs      | Core to the experiment                      | **Critical** |
| No tooling to enumerate all valid (encoder, combiner, decoder) combos for a schema | Can't define the search space               | **Critical** |
| No config diversity / coverage metrics                                             | May oversample some regions of search space | **Medium**   |
| No config pre-flight validator (does this config even compile for this dataset?)   | Wasted compute on invalid configs           | **High**     |
| No resource estimation per (config, dataset)                                       | Can't pre-allocate compute                  | **Medium**   |

#### Experiment Infrastructure Gaps

| Gap                                                                | Impact                                    | Priority     |
| ------------------------------------------------------------------ | ----------------------------------------- | ------------ |
| No results database — per-experiment dirs only                     | Can't compare 50k runs                    | **Critical** |
| No meta-scheduler for dataset × config matrix                      | No way to orchestrate the full experiment | **Critical** |
| No failure recovery / retry logic                                  | Any crash at scale is fatal               | **High**     |
| No experiment state tracking (queued/running/done/failed per cell) | Can't resume interrupted runs             | **High**     |
| No cross-dataset results comparison infrastructure                 | Analysis impossible                       | **Critical** |
| No compute cost estimation tool                                    | Can't plan cloud spend                    | **Medium**   |

______________________________________________________________________

## 3. Phased Implementation Plan

### Phase 0 — Foundation: Dataset Curation Pipeline (3–4 weeks)

#### 0.1 OpenML Integration

**Goal:** Download any OpenML task + dataset programmatically.

```
ludwig/datasets/loaders/openml_loader.py
```

Tasks:

- Install `openml` Python SDK
- Create `OpenMLLoader(DatasetLoader)` that accepts `openml_task_id`
- Load via `openml.tasks.get_task(task_id)` → get dataset + target column + predefined splits
- Store as Parquet in `~/.ludwig_cache/openml_{task_id}/`
- Register priority suites: **OpenML-CC18** (72 datasets), **OpenML-CTR23** (35 regression), **suite 269** (classification), **suite 271** (regression)
- Expose as `from ludwig.datasets import openml_cc18` → returns list of loaders

Deliverables:

- `OpenMLLoader` class
- `openml_suite_loaders(suite_id: int) -> list[DatasetLoader]`
- 107+ datasets accessible

#### 0.2 Kaggle Dataset Scraper

**Goal:** Build a pipeline to discover, filter, and ingest ML-ready Kaggle tabular datasets.

```
scripts/kaggle_discovery/
├── scrape_kaggle.py          # Paginate kaggle API, collect metadata
├── filter_kaggle.py          # Apply quality filters
├── download_kaggle.py        # Batch download + cache
└── kaggle_registry.json      # Persisted dataset catalog
```

Filters to apply (in order):

1. `file_type=csv` (tabular format)
1. `min_size=50000` (≥50KB — eliminates toy datasets)
1. `max_size=500_000_000` (≤500MB — avoid multi-GB)
1. Tags: `14101` (Tabular Data) OR `1241` (Classification) OR `5593` (Regression)
1. `license` in `{cc0-1.0, cc-by-4.0, odc-odbl}` (permissive licenses only)
1. Post-download quality filters:
   - At least 500 rows
   - At least 3 feature columns
   - At least 1 unambiguous target column (see §0.3)
   - No more than 30% missing values overall
   - Not pure image/audio/text (≥1 numeric or categorical column)

**API pagination pattern:**

```python
from kaggle.api.kaggle_api_extended import KaggleApiExtended

api = KaggleApiExtended()
api.authenticate()
for page in range(1, 501):  # 500 pages × 20/page = 10,000 candidates
    datasets = api.dataset_list(file_type="csv", tags="14101", page=page)
    if not datasets:
        break
    for ds in datasets:
        catalog.append(
            {
                "ref": ds.ref,
                "title": ds.title,
                "size": ds.totalBytes,
                "downloads": ds.downloadCount,
                "votes": ds.voteCount,
                "updated": ds.lastUpdated,
            }
        )
```

Target: curate **300–500 ML-ready Kaggle tabular datasets**.

#### 0.3 Auto Target Column Detection

**Goal:** For a raw CSV with no specified target, infer the most likely target column.

```
ludwig/automl/target_detection.py
```

Heuristics (ordered by confidence):

1. Column named exactly `target`, `label`, `y`, `output`, `class`, `outcome`, `result` → high confidence
1. Last column in the CSV (common convention) → medium confidence
1. Lowest-cardinality non-binary column (likely class label) → medium confidence
1. Column with highest inter-feature correlation sum (could be response) → low confidence
1. Columns that look like IDs (all-unique string or sequential int) → exclude
1. Boolean / binary column with ~50/50 split → candidate

Return: `(column_name, confidence_score, task_type)` where `task_type ∈ {binary, multiclass, regression}`.

#### 0.4 Dataset Quality Filter

```
ludwig/utils/dataset_quality.py
```

Checks to run after download:

- **Target leakage detection**: any feature with correlation > 0.99 with target → flag
- **Constant columns**: variance = 0 → drop
- **Near-duplicate columns**: pairwise correlation > 0.99 → flag
- **ID columns**: detect sequential integers, UUIDs, hashes → exclude as input
- **Data type compatibility**: ensure Ludwig can parse each column (no mixed-type chaos)
- **Class imbalance**: log minority class ratio (warn if < 1%)
- **Minimum viable size**: at least 200 rows × 3 features after cleaning

Output: `DatasetQualityReport` (pass/fail/warn per check, recommended feature schema)

#### 0.5 Dataset Registry Expansion

Extend `ludwig/datasets/configs/` YAML format to support OpenML task IDs:

```yaml
# openml_adult.yaml
version: 1.0
name: openml_adult
openml_task_id: 7592
description: "UCI Adult Census Income classification (OpenML task 7592)"
output_features:
  - name: class
    type: binary
```

______________________________________________________________________

### Phase 1 — Config Generation Engine (2–3 weeks)

#### 1.1 Schema-Aware Config Enumerator

**Goal:** Given a `FeatureSchema` (list of `(name, type)` pairs), enumerate all valid Ludwig config combinations.

```
ludwig/automl/config_enumerator.py
```

```python
@dataclass
class FeatureSchema:
    input_features: list[tuple[str, str]]  # (name, type)
    output_features: list[tuple[str, str]]  # (name, type)


def enumerate_valid_configs(schema: FeatureSchema) -> list[ModelConfig]:
    """
    Returns all structurally valid Ludwig ModelConfigs for the given schema.
    Validity rules:
    - Each encoder must be compatible with its input feature type
    - Each decoder must be compatible with its output feature type
    - Combiner must be compatible with the input feature set
    """
```

Compatibility rules to encode (from Ludwig's schema validation):

- `tabnet` combiner: only for schemas with ≥2 tabular features (number/binary/category)
- `transformer` combiner: works for any schema
- `concat` combiner: universal fallback
- Text encoders (bert, etc.): only for `text` input features
- Image encoders (stacked_cnn, resnet): only for `image` input features
- Timeseries encoders (nbeats, patchtst): only for `timeseries` input features
- `generator` decoder: only for `text` output
- `regressor` decoder: only for `number` output
- `classifier` decoder: only for `binary` or `category` output

#### 1.2 Config Sampler (100 Configs per Dataset)

**Goal:** Sample 100 diverse configs that maximally cover the valid config space.

```
ludwig/automl/config_sampler.py
```

Strategy: **Stratified sampling** across the key axes of variation:

```
Axis 1: Combiner (16 options)
  - Always include: tabnet (if applicable), transformer, concat, tab_transformer, ft_transformer
  - Coverage: at least 5 combiners represented across 100 configs

Axis 2: Encoder per input feature type
  - For tabular (category/number/binary): onehot, passthrough, dense, sparse, target
  - For text: bert, distilbert, roberta, xlnet, tf_idf, embed
  - For image: stacked_cnn, resnet18, efficientnet_b0, vit
  - For timeseries: nbeats, patchtst, mamba2

Axis 3: Decoder
  - For binary output: mlp_classifier, regressor
  - For category output: classifier, category_extractor
  - For number output: regressor, projector
  - For text output: generator, tagger, transformer_generator

Axis 4: Trainer hyperparameters (5 grid points per param)
  - learning_rate: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
  - batch_size: [64, 128, 256, 512, 1024]
  - num_epochs: [10, 25, 50]

Axis 5: Regularization
  - dropout: [0.0, 0.1, 0.3]
  - weight_decay: [0.0, 1e-5, 1e-4]
```

Sampling algorithm: **Sobol quasi-random sequence** over the joint space → ensures better coverage than pure random, avoids grid explosion.

Each of the 100 configs should be:

- Structurally valid (passes Ludwig schema validation)
- Diverse (no two configs identical on all axes)
- Computationally feasible (estimated training time < 30 min on a single GPU for the given dataset size)

#### 1.3 Config Pre-flight Validator

**Goal:** Before spending compute, verify a config will not crash.

```
ludwig/automl/config_validator.py

def validate_config_for_dataset(
    config: ModelConfig,
    dataset: pd.DataFrame,
    schema: FeatureSchema,
) -> ValidationResult:
    """
    Returns (is_valid, failure_reason) without actually training.
    Checks:
    - Schema compatibility (encoder/combiner/decoder types)
    - Dataset size vs. batch size (need at least 2 batches)
    - Memory estimate (rough GPU VRAM check)
    - Required columns exist in dataset
    """
```

#### 1.4 Config Serialization

Store the 100 configs for each dataset as:

```
benchmark/configs/{dataset_name}/
├── configs.jsonl           # 100 configs, one per line
├── schema.json             # Feature schema used
├── sampler_seed.json       # For reproducibility
└── validity_report.json    # Pre-flight results
```

______________________________________________________________________

### Phase 2 — Experiment Infrastructure (3–4 weeks)

#### 2.1 Results Database

**Design:** DuckDB over Parquet files — zero-server, analytical queries, cross-platform.

```
benchmark/results/
├── runs/
│   ├── {dataset}_{config_hash}_{seed}.parquet    # Per-run results
│   └── ...
├── runs_index.parquet          # Materialized index of all runs
├── predictions/                # Optional: raw predictions per run
│   └── {run_id}.parquet
└── analysis/
    ├── per_dataset_rankings.parquet
    ├── per_config_rankings.parquet
    └── overall_leaderboard.parquet
```

Schema for `runs_index.parquet`:

```python
{
    "run_id": str,  # UUID
    "dataset_name": str,
    "dataset_source": str,  # "openml", "kaggle", "ludwig_builtin"
    "dataset_n_rows": int,
    "dataset_n_features": int,
    "config_hash": str,  # SHA256 of config JSON
    "combiner": str,
    "input_encoders": str,  # JSON list
    "output_decoder": str,
    "learning_rate": float,
    "batch_size": int,
    "n_epochs": int,
    "seed": int,
    "status": str,  # "queued" / "running" / "done" / "failed"
    "start_time": datetime,
    "end_time": datetime,
    "wall_seconds": float,
    "gpu_type": str,
    "primary_metric": str,
    "primary_metric_value": float,
    "secondary_metrics": str,  # JSON dict
    "error_message": str,  # null if done
    "checkpoint_path": str,
}
```

Queries to support:

```sql
-- Best config per dataset
SELECT dataset_name, config_hash, MAX(primary_metric_value) AS best
FROM runs_index WHERE status='done'
GROUP BY dataset_name
ORDER BY dataset_name;

-- Which combiner wins most often
SELECT combiner, COUNT(*) AS wins FROM (
  SELECT DISTINCT ON (dataset_name) combiner, primary_metric_value
  FROM runs_index WHERE status='done'
  ORDER BY dataset_name, primary_metric_value DESC
) GROUP BY combiner;

-- Datasets where Ludwig beats AutoGluon baseline
SELECT r.dataset_name, r.primary_metric_value AS ludwig_best, b.automl_best
FROM runs_index r JOIN baselines b USING(dataset_name)
WHERE r.primary_metric_value > b.automl_best AND r.status='done';
```

**Implementation:** `benchmark/db.py` wrapping DuckDB + Parquet append.

#### 2.2 Meta-Scheduler

**Goal:** Orchestrate the N-datasets × M-configs matrix as a distributed job queue.

```
benchmark/scheduler.py
```

Architecture:

```
BenchmarkScheduler
├── JobQueue (Redis or SQLite-backed)
│   └── Job: {dataset_name, config_path, seed, priority, status}
├── RayClusterManager
│   └── submit_job(job) → ray.remote(run_experiment)(...)
├── ResultCollector
│   └── on_job_complete(job_result) → write to DuckDB
└── FailureHandler
    └── on_job_fail(job, error) → retry up to 3× with exponential backoff
```

Key features:

- **Priority queue**: start with small datasets first (faster iteration)
- **Resume from checkpoint**: on restart, skip `status=done` jobs
- **Resource-aware scheduling**: don't submit more jobs than available GPU slots
- **Dead-letter queue**: jobs that fail >3× go to `failed_jobs.jsonl` for manual review
- **Progress dashboard**: `benchmark/dashboard.py` — live terminal UI (Rich library)

#### 2.3 Single-Experiment Runner

```
benchmark/runner.py

def run_experiment(
    dataset_name: str,
    dataset_source: str,   # "openml" | "kaggle" | "ludwig"
    config_path: str,
    output_dir: str,
    seed: int = 42,
    gpu_id: int = 0,
    time_limit_s: int = 1800,  # 30 min default
) -> RunResult:
    """
    1. Load dataset (from cache or download)
    2. Load config (from configs.jsonl)
    3. Run ludwig.train() with that config
    4. Evaluate on test set
    5. Return RunResult (metrics + metadata)
    """
```

Key behaviors:

- Hard time limit via `signal.alarm` (or `ray`'s `runtime_env` timeout)
- OOM handling: catch `RuntimeError: CUDA out of memory` → retry with smaller batch size
- NaN loss detection: after first epoch, if loss is NaN → mark failed, don't waste budget
- Progress logging: write partial results every epoch to DuckDB (allows analysis of failed runs)

#### 2.4 Baseline Comparison

To make results meaningful, we need baselines for each dataset:

```
benchmark/baselines/
├── run_autogluon.py     # AutoGluon-Tabular 4-hour budget
├── run_xgboost.py       # XGBoost default + basic HPO
├── run_lightgbm.py      # LightGBM default
└── baselines.parquet    # Precomputed or loaded from TabRepo/AMLB
```

**Shortcut:** Use TabRepo's precomputed results for 200 OpenML datasets — 219,136 CPU-hours of baselines are already computed and publicly available. We just need to join on `openml_task_id`.

#### 2.5 Monitoring Dashboard

```
benchmark/dashboard.py
```

Live terminal display (Rich):

```
Ludwig Mega-Benchmark — Progress
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Datasets:    342 / 500   (68.4%)
Configs:     23,540 / 50,000  (47.1%)
Queued:      12,450     Running: 16    Done: 23,524    Failed: 10
GPU hours:   1,847 / ~4,000 (est.)     ETA: 6d 14h
Top combiner so far:    tabnet (wins 34%)
Top encoder (text):     bert (wins 52%)
Slowest datasets:       criteo_kaggle (avg 28 min/config)
```

______________________________________________________________________

### Phase 3 — Pilot Run (1–2 weeks)

**Target:** 10 datasets × 10 configs = 100 experiments

**Dataset selection:**

- 5 from OpenML-CC18: adult, higgs, covertype, jannis, numerai28.6
- 5 from Ludwig built-ins: titanic, adult_census_income, california_housing, ibm_employee_attrition, otto_product_classification

**Config selection:** 2 per combiner × 5 combiners (tabnet, concat, transformer, tab_transformer, ft_transformer)

**Infrastructure:** Single machine with 1–2 GPUs

**Success criteria:**

- All 100 runs complete (or fail gracefully with informative errors)
- Results written to DuckDB
- Results are reasonable (no combinator always wins on all datasets)
- Wall time ≤ 4 hours total

______________________________________________________________________

### Phase 4 — Medium Scale (2–4 weeks)

**Target:** 100 datasets × 50 configs = 5,000 experiments

**Dataset selection:**

- 72 from OpenML-CC18
- 28 from Kaggle (well-known: titanic, house prices, otto, porto, etc.)

**Config selection:** Full 50-config sampler from Phase 1

**Infrastructure:** 4–8 GPU nodes (AWS `g4dn.xlarge` × 8 = ~$12/hr), Ray cluster

**Compute estimate:**

- Average 15 min/config on T4 GPU
- 5,000 configs × 15 min = 1,250 GPU-hours
- At 8 parallel GPUs: ~156 hours wall time → ~1 week
- Cost: ~$500–800 on spot instances

**Success criteria:**

- > 90% of runs complete (≤10% failure rate)
- Results show meaningful variation (some configs win on some datasets, not universal winner)
- Results align with known baselines (TabNet should win on some tabular datasets)

______________________________________________________________________

### Phase 5 — The Mega Run

**Target:** 400–500 datasets × 100 configs = 40,000–50,000 experiments

**Dataset breakdown:**

- 107 from OpenML (CC18 × 72 + CTR23 × 35)
- 200 from curated Kaggle tabular datasets (Phase 0.2 output)
- 93 from Ludwig built-ins (all existing)
- 100 from HuggingFace datasets (tabular classification/regression)
- Total: ~500 datasets

**Config distribution per dataset (100 configs):**

- Tabular-only datasets (number/binary/category inputs):
  - 25 TabNet variants (varying size, num_steps, sparsity)
  - 20 Concat FC variants (varying layers, output_size, dropout)
  - 20 Transformer variants (varying num_heads, depth, ffn_size)
  - 15 Tab-Transformer variants
  - 10 FT-Transformer variants
  - 10 TabPFN configs (few hyperparams)
- Text-containing datasets:
  - 30 BERT/DistilBERT/RoBERTa encoder × concat combiner
  - 20 TF-IDF encoder × tabnet/concat combiners
  - 20 embed encoder variants
  - 30 mixed (text + tabular features)
- Image-containing datasets:
  - 30 StackedCNN variants
  - 30 ResNet18/34 variants
  - 20 EfficientNet variants
  - 20 ViT variants

**Infrastructure:**

- Ray cluster: 20 nodes × 8 GPUs (A10G or T4) = 160 GPUs
- At 15 min avg/config: 50,000 × 15 min / 160 = ~78 hours wall time
- Compute: ~7,500 GPU-hours
- Estimated cost: ~$7,500–15,000 on spot instances (A10G ~$1/hr/GPU)

**Fault tolerance:**

- Spot instance preemption recovery (Ray handles via checkpoint)
- Max 3 retries per failed job
- Results continuously flushed to S3-backed DuckDB

______________________________________________________________________

### Phase 6 — Analysis and Publication

#### 6.1 Analysis Notebooks

```
notebooks/
├── 01_dataset_statistics.ipynb           # Dataset corpus overview
├── 02_config_performance_overview.ipynb  # Which configs win overall
├── 03_combiner_comparison.ipynb          # Combiner head-to-head
├── 04_encoder_comparison.ipynb           # Encoder head-to-head by feature type
├── 05_vs_tabular_baselines.ipynb         # Ludwig vs XGBoost/LightGBM/AutoGluon
├── 06_scaling_analysis.ipynb             # Dataset size vs. best architecture
├── 07_metafeature_analysis.ipynb         # What dataset properties predict winner
└── 08_leaderboard.ipynb                  # Overall rankings
```

Key questions to answer:

1. On tabular data (no text/image), does Ludwig match AutoGluon/LightGBM?
1. On multimodal data, how does Ludwig compare to AutoGluon-Multimodal?
1. Which combiner is most consistently best across dataset types?
1. Does dataset size predict the best architecture family?
1. How much does encoder choice matter vs. combiner choice vs. hyperparameters?
1. Is there a "meta-learned" config that works well everywhere (a universal prior)?

#### 6.2 Meta-Learning Over Results

Using the results as training data for a meta-learning system:

- Input: dataset meta-features (n_rows, n_cols, feature types, class imbalance, etc.)
- Output: predicted best config hash (or config parameters)
- Models to try: GradientBoostingClassifier, TabPFN over the meta-features

This could power a much better `auto_train()` that picks architectures based on dataset characteristics.

#### 6.3 Paper / Report

Target venue: **NeurIPS 2026 Datasets and Benchmarks track** (deadline ~June 2026).

Title candidates:

- "Ludwig-Bench: The Largest Deep Learning AutoML Benchmark for Tabular and Multimodal Data"
- "Scaling Ludwig: A 50,000-Run Benchmark of Deep Learning Architectures on Tabular Data"

Key contributions:

1. The benchmark itself (dataset + config corpus, results DB — all public)
1. Analysis of which DL architectures generalize across dataset types
1. A meta-learned config selector trained on benchmark results
1. A living leaderboard (new datasets/configs can be added)

______________________________________________________________________

## 4. Technical Debt and Improvements to Ludwig Core

### 4.1 Must Fix Before Mega Run

| Issue                                             | Fix                                                                                           | Effort |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------ |
| No multi-dataset parallelism in hyperopt executor | Add a `BenchmarkExecutor` that maps (dataset, config) pairs to Ray tasks                      | Medium |
| `auto_train()` generates only 1 config            | Separate config generation from execution; expose `generate_config_candidates(schema, n=100)` | Medium |
| No pre-flight config validation                   | `validate_config_for_dataset(config, df)` without training                                    | Small  |
| OpenML task loading not supported                 | `OpenMLLoader` class                                                                          | Small  |
| No auto target detection                          | `detect_target_column(df) -> (name, task_type)`                                               | Small  |
| Hyperopt results stored in local dirs only        | Add `BenchmarkResultsWriter` that writes to DuckDB                                            | Medium |
| No dataset quality filtering                      | `DatasetQualityFilter` class                                                                  | Small  |

### 4.2 Nice to Have

- **Faster config validation**: Use Ludwig's Pydantic schema to validate without constructing model objects
- **Config fingerprinting**: Deterministic hash of a `ModelConfig` for deduplication
- **Time-series split support**: Properly handle datasets with temporal structure
- **Stratified sampling in config generator**: Ensure diverse coverage across combiner axes
- **Incremental feature importance**: Per-run SHAP or permutation importance stored alongside metrics

______________________________________________________________________

## 5. Compute and Cost Estimates

### Single Machine (Development / Pilot)

| Stage            | Configs | Avg Time | Wall Time | GPU     |
| ---------------- | ------- | -------- | --------- | ------- |
| Pilot (Phase 3)  | 100     | 15 min   | 25 hrs    | 1× A100 |
| Medium (Phase 4) | 5,000   | 15 min   | 6.5 days  | 8× T4   |

### Cloud Cluster (Mega Run)

| Instance       | vCPU | RAM    | GPU             | Spot Price | Throughput    |
| -------------- | ---- | ------ | --------------- | ---------- | ------------- |
| `g4dn.xlarge`  | 4    | 16 GB  | 1× T4 (16 GB)   | ~$0.15/hr  | 4 configs/hr  |
| `g4dn.4xlarge` | 16   | 64 GB  | 1× T4 (16 GB)   | ~$0.52/hr  | 4 configs/hr  |
| `g5.xlarge`    | 4    | 16 GB  | 1× A10G (24 GB) | ~$0.50/hr  | 4 configs/hr  |
| `g5.12xlarge`  | 48   | 192 GB | 4× A10G         | ~$3.00/hr  | 16 configs/hr |

**Recommended setup:**

- 20× `g5.12xlarge` nodes → 80× A10G GPUs → 320 concurrent configs
- 50,000 configs ÷ 320 parallel ÷ 4 configs/GPU/hr = **~39 hours wall time**
- 50,000 × 0.25 hrs × 80 GPUs = **1,000 GPU-hours**
- At $3.00/hr spot × 20 nodes × 39 hrs = **~$2,340 compute cost** (spot pricing)

### Storage

- Per-run result Parquet: ~10 KB × 50,000 = 500 MB (trivial)
- Per-run model checkpoint: ~100 MB × 50,000 = 5 TB (optional; only keep best per dataset)
- Recommendation: store checkpoints for top-5 configs per dataset only → ~200 GB

______________________________________________________________________

## 6. Timeline

```
Month 1 (May 2026):
  Week 1-2: Phase 0.1 (OpenML integration) + Phase 0.3 (target detection)
  Week 3-4: Phase 0.2 (Kaggle scraper) + Phase 0.4 (quality filter)

Month 2 (June 2026):
  Week 1-2: Phase 1.1-1.2 (config enumerator + sampler)
  Week 3-4: Phase 1.3-1.4 (validator + serialization)

Month 3 (July 2026):
  Week 1-2: Phase 2.1-2.2 (results DB + meta-scheduler)
  Week 3: Phase 2.3-2.4 (runner + baselines)
  Week 4: Phase 3 (pilot run — 100 experiments)

Month 4 (August 2026):
  Week 1-2: Fix issues from pilot, tune infrastructure
  Week 3-4: Phase 4 (medium run — 5,000 experiments)

Month 5 (September 2026):
  Week 1-2: Fix issues from medium run
  Week 3-4: Phase 5 (mega run — 50,000 experiments)

Month 6 (October-November 2026):
  Weeks 1-4: Phase 6 (analysis, meta-learning, paper writing)

Target: NeurIPS 2026 Datasets & Benchmarks submission (deadline ~June 2026)
  → adjust timeline: Phase 5 may need to happen in parallel with early analysis
```

______________________________________________________________________

## 7. Risks and Mitigations

| Risk                                                   | Likelihood | Impact | Mitigation                                                                      |
| ------------------------------------------------------ | ---------- | ------ | ------------------------------------------------------------------------------- |
| Kaggle API rate limiting blocks dataset scraping       | Medium     | High   | Paginate slowly (1 req/sec), cache responses, use OpenML as primary corpus      |
| Many Kaggle datasets have ambiguous target columns     | High       | Medium | Auto-detection (§0.3) + human review of uncertain cases                         |
| 30%+ of configs OOM on given dataset                   | Medium     | High   | Pre-flight memory estimation, conservative batch size defaults, OOM catch-retry |
| Spot instances preempted mid-run                       | High       | Low    | Ray Train checkpointing every epoch; resume on restart                          |
| Results not reproducible across hardware               | Medium     | High   | Pin PyTorch/CUDA versions, log GPU type per run, use fixed seeds                |
| Ludwig schema validation rejects valid-looking configs | Medium     | Medium | Pre-flight validator; unit test all combiners × feature type combos             |
| Benchmark costs exceed budget                          | Low        | High   | Monitor spend hourly; cap at configurable total GPU-hours                       |
| Legal issues with Kaggle dataset licenses              | Low        | High   | Only use `cc0`, `cc-by-4.0`, `odc-odbl` licensed datasets                       |

______________________________________________________________________

## 8. Defining "Largest Ever"

To claim the title, we need:

| Metric                             | Target              | TabArena (current leader)                                |
| ---------------------------------- | ------------------- | -------------------------------------------------------- |
| Total model training runs          | **50,000**          | ~25 million (but 16 models, not 50,000 distinct configs) |
| Distinct architectures explored    | **100 per dataset** | ≤200 HP configs but only 16 model families               |
| Datasets covered                   | **400–500**         | 51 (from 1,053 candidates)                               |
| Deep learning configs specifically | **50,000**          | ~4,000 (TabNet, SAINT, TabR, TabM, TabPFN, TabDPT)       |
| Multimodal experiments             | **>1,000**          | 0 (TabArena is tabular-only)                             |
| Distinct combiners tested          | **16**              | Not applicable                                           |

**The claim:** "The largest systematic evaluation of deep learning architectures for tabular and multimodal data, covering N architectures across M datasets — 10× more distinct deep learning configurations than any prior benchmark."

Note: TabArena's "25 million" counts individual sample-level predictions; our 50,000 counts full model training runs. Both framings are valid and complementary. We should present our metric as "model training runs" to be precise.

______________________________________________________________________

## 9. Immediate Next Steps (This Week)

1. **Audit Ludwig's existing automl type inference** — confirm `ludwig/utils/automl/type_inference.py` is exposed for standalone use
1. **Install `openml` package** and prototype `OpenMLLoader` for one task (adult income, task 7592)
1. **Run Kaggle API scraper prototype** — page through 50 pages, verify filter logic works
1. **Build config sampler prototype** — for the titanic schema, generate 10 diverse configs, verify they all pass `ModelConfig` validation
1. **Set up DuckDB results schema** — create the `runs_index.parquet` schema and test append/query

______________________________________________________________________

*This plan was written based on analysis of Ludwig v0.16.x codebase (May 2026) and review of the following benchmarks: TabArena (NeurIPS 2025), TabRepo (2024), TabZilla (NeurIPS 2023), AMLB (JMLR 2024), and related work.*
