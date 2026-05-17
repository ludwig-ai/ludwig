# Ludwig Codebase Improvement Plan

Generated: 2026-05-16. Based on a thorough review of the full codebase (110k lines, ~400 files).

---

## Executive Summary

Ludwig is **architecturally sound at the macro level** — the Backend abstraction, modular encoder/decoder registries, and schema-driven config system are genuinely well-designed. The problem is at the **meso level**: a handful of files have become god objects that grow without bound (`preprocessing.py` 2407 lines, `api.py` 2237 lines, `trainer.py` 1766 lines), there are 208 untyped `Any` fields hiding data contracts, 112 bare `except Exception:` handlers silencing failures, and 172 TODOs indicating unresolved design decisions. The single most impactful structural fix is decomposing the 18-class `DataFormatPreprocessor` hierarchy into a reader-strategy pattern (~900 lines of duplication removed). The single most impactful production fix is auditing the 112 bare `except Exception:` handlers.

---

## Critical Issues (Must Fix)

### C1 — `FatherPreprocessor` typo (`preprocessing.py:689`)
The Feather format preprocessor is named `FatherPregressor`. It is mapped to `FEATHER_FORMATS` at line 1149. Any contributor looking for the Feather reader would never find it.
- **Fix:** Rename to `FeatherPreprocessor` (class + format map).

### C2 — Duplicate `keys()` on `TrainingStats` (`api.py:150,168`)
`TrainingStats.keys()` is defined twice. The second definition silently overrides the first; suppressed with `# noqa: F811`. Breaks the Mapping protocol contract.
- **Fix:** Delete lines 150-153 (the first definition).

### C3 — 18 nearly-identical `DataFormatPreprocessor` subclasses (`preprocessing.py:186-1046`)
Every subclass (`DictPreprocessor`, `CSVPreprocessor`, `JSONPreprocessor`, `ParquetPreprocessor`, ...) implements `preprocess_for_training()`, `preprocess_for_prediction()`, `prepare_processed_data()` with bodies that differ only in the pandas read call. ~900 lines of structural duplication.
- **Fix:** See PR-3.

### C4 — Silent failure on missing files (`backend/base.py:191-212`)
`LocalPreprocessingMixin.read_binary_files()` passes `None` paths to `map_fn` without error. Datasets silently drop samples at scale.
- **Fix:** Add `None` guard before calling `map_fn`; raise `ValueError` with path context.

### C5 — 112 bare `except Exception:` handlers throughout codebase
Examples: `except Exception: return None` in `strings_utils.py:138`, `except Exception: print("FAILURE")` in `check.py:32`. Makes production debugging impossible.
- **Fix:** See PR-7.

### C6 — OOM footgun in `collect_activations()` (`collect.py:186`)
Marked `# TODO -> Fix OOM on large models e.g. llama 3 8B`. Loads all activations for all samples into RAM at once. Will OOM on any LLM with >10k samples.
- **Fix:** Stream activations in chunks; write intermediate results to disk.

---

## Major Issues (Should Fix)

### M1 — `dict[str, Any]` type aliases hide all data contracts (`types.py:5-47`)
All public typedefs (`FeatureConfigDict`, `ModelConfigDict`, `TrainingSetMetadataDict`, etc.) are `dict[str, Any]`. 208 total `Any` usages. Disables IDE completion and static analysis.
- **Fix:** See PR-1.

### M2 — Trainer is a 1766-line god object (`trainer.py:104-1766`)
`Trainer` inherits from 4 mixins + `BaseTrainer`. `__init__` has 24 parameters. `train_loop()` is ~240 lines mixing batching, backprop, gradient accumulation, metrics, checkpointing, early stopping.
- **Fix:** See PR-5, PR-6.

### M3 — `LudwigModel` is a 2237-line god object (`api.py:160-2237`)
Mixes model lifecycle, serialization, experiment logging, hyperopt, and serving.
- **Fix:** See PR-8.

### M4 — Feature preprocessing modules have no shared base
`_ImagePreprocessing`, `_AudioPreprocessing`, `_CategoryPreprocessing` etc. each re-implement missing value handling, dtype casting, reshaping with no common base class.
- **Fix:** See PR-4.

### M5 — `create_passthrough_input_feature()` is an invisible feature type (`base_feature.py:648-703`)
Defines an inline class that bypasses the feature registry entirely. Undiscoverable.
- **Fix:** See PR-9.

### M6 — 92 suppressed type errors (`# type: ignore`, `# noqa`)
Including bugs: `api.py:288` has `# type: ignore [assignment]` because `self.config_fp = None` when type expects `str` — that's an uninitialised-state bug.
- **Fix:** Audit all 92; fix root causes.

### M7 — Test coverage gaps on critical paths
- `preprocessing.py` — no unit tests for individual `DataFormatPreprocessor` subclasses
- `backend/ray.py:816` — `BatchInferModel` inner class untested
- `trainers/trainer_dpo.py` — `KTOTrainer`, `ORPOTrainer`, `GRPOTrainer` have no unit tests
- `collect.py` — zero unit tests

### M8 — Stale TODOs that are actually bugs
- `trainer.py:302` — "loading an existing model loses metric values" — known data loss on resume
- `models/base.py:147` — "Remove dummy implementation" — dummy property returns wrong values
- `api.py:1980` — model type check duplicated between LLM and ECD paths

---

## Minor Issues

### Naming (violating "naming things" rules)

| File:Line | Problem | Fix |
|-----------|---------|-----|
| `preprocessing.py:689` | `FatherPreprocessor` — typo for Feather | `FeatherPreprocessor` |
| `backend/base.py:181` | `LocalPreprocessingMixin` — also handles binary reads | `LocalDataProcessingMixin` |
| `features/base_feature.py:57` | `BaseFeatureMixin` — vague, has state | `FeaturePreprocessingMixin` |
| `models/base.py:125` | `ModuleWrapper` — says nothing about purpose | `NonPropertyModuleWrapper` |
| `trainers/trainer_llm.py:44` | `NoneTrainer` — confusing name for inference-only | `InferenceOnlyTrainer` |
| `data/cache/manager.py:101` | `CacheManager` — generic Manager anti-pattern | `PreprocessedDataCache` |

### Type Hints
- `backend/base.py:79` — `capabilities: dict[str, Any]` → `dict[str, bool]`
- `api.py:145-147` — `TrainingStats` fields → `dict[str, float]`
- `features/base_feature.py` — 12 methods missing return type annotations
- `encoders/text_encoders.py:184` — abstract method `get_hf_config_param_names` never enforced

### Docstrings
- `features/base_feature.py:100` — `add_feature_data()` doesn't describe `proc_df` contract
- `models/ecd.py:145` — `forward()` doesn't explain `targets` in train vs. predict mode
- `data/preprocessing.py:143` — `DataFormatPreprocessor` has no docstring for the 3-method contract

### Performance
- `preprocessing.py:207-214` — converts `dataset` to `pd.DataFrame` 3× in same function
- `utils/data_utils.py:452` — `hash_dict()` runs `pickle.dumps()` + SHA256 on every call; not cached
- `features/base_feature.py:178` — `create_sample_input()` generates random tensors every call

### Magic Constants
- `data/lazy_utils.py:29` — `min(16, (os.cpu_count() or 4) + 4)` — why `+4`? why cap 16? Add comment.
- `features/image_feature.py:98-99` — ImageNet1K mean/std hardcoded instead of from torchvision

### Dead Code
- `features/base_feature.py:648-703` — `create_passthrough_input_feature()` inline class factory
- `utils/visualization_utils.py` — 1568 lines of custom plotting duplicating pandas/plotly

---

## Persona Verdicts

### ML Engineer (Production Pipelines)
**HIGH RISK** — do not deploy without fixing C4, C5, C6. Silent failures in `read_binary_files()` (C4) silently shrink datasets at scale. 112 bare `except Exception:` (C5) mean production tracebacks are useless. The OOM footgun in `collect.py` (C6) hits every practitioner who runs feature analysis on an LLM. The preprocessing monolith (2407 lines, 18 classes) makes debugging data pipelines require holding an enormous mental model. The trainer mixes concerns so tightly that adding distributed debugging hooks requires touching 6 different mixins.

### ML Researcher (Running Experiments)
**MODERATE** — good for prototyping, risky for long-running experiments. The declarative YAML config and pydantic schema validation are the right idea, well-executed. However: the known metric-loss-on-resume bug (M8, `trainer.py:302`) is a real reproducibility hazard for multi-day training runs. The HuggingFace encoder schema (`schema/encoders/text_encoders.py`: 2714 lines) is so large that understanding available params for a given model requires reading 100+ lines. Hyperopt is tightly coupled to trainer internals, making custom search spaces fragile across versions.

### Open Source Contributor (First PR)
**UNWELCOMING** — needs a contributor guide and smaller files. Adding a new feature type requires understanding: (1) `BaseFeatureMixin` vs `InputFeature`/`OutputFeature` split, (2) the schema-vs-feature-module duality (every feature has a matching schema class in `ludwig/schema/features/`), (3) the inner preprocessing module pattern, (4) the encoder/decoder registry. None of this is documented in one place. Feature files are enormous (image: 1378 lines, 66 methods; audio: 675 lines). `create_passthrough_input_feature()` is an invisible feature type that bypasses the registry — a contributor following the normal pattern would never know it exists.

### Social Media ML Reader (HN/Reddit/X)
**MIXED** — impressive scope, cringe-worthy internals. The feature set is genuinely impressive (600+ datasets, 50+ encoders, Ray distributed, LLM fine-tuning, multimodal). But: `preprocessing.py` has 18 classes with copy-pasted method bodies. `api.py` is 2237 lines. `FatherPreprocessor` (a Feather reader named "Father") has been in production. The `dict[str, Any]` typedefs look like hastily-migrated Python 2 code. 172 TODOs suggest active development paralysis. The architecture deserves better than the execution.

---

## Improvement Plan (Ordered PRs)

### Phase 0 — Quick Wins (1-2 days, zero risk)

**PR-0a: Fix typos + silent bugs** (S)
- `FatherPreprocessor` → `FeatherPreprocessor` (`preprocessing.py:689,1149`)
- Remove duplicate `keys()` from `TrainingStats` (`api.py:150-153`)
- Add `None` guard in `read_binary_files()` (`backend/base.py:207`)

**PR-0b: Fix TODO bugs** (S)
- Fix metric loss on resume (`trainer.py:302`) — reload metrics from checkpoint
- Remove fake `input_shape` dummy property (`models/base.py:147`)
- Fix `check.py:32` silent failure — add `logger.exception()`

---

### Phase 1 — Type System (1 week)

**PR-1: TypedDict for data contracts** (L)
Replace `dict[str, Any]` aliases in `types.py` with `TypedDict` subclasses. Update callsites. Run mypy; fix revealed type errors.
- `FeatureConfigDict`, `TrainingSetMetadataDict`, `FeatureMetadataDict` etc.
- Files: `types.py`, `api.py`, `features/base_feature.py`, `data/preprocessing.py`
- Impact: ~50 latent bugs caught by mypy; IDE completion for configs

**PR-2: Backend capabilities as frozen dataclass** (S)
```python
@dataclass(frozen=True)
class BackendCapabilities:
    distributed: bool = False
    hyperopt: bool = False
    async_execution: bool = False
```
Replace all string-key capability lookups.
- Files: `backend/base.py`, `backend/ray.py`, `backend/local.py`

---

### Phase 2 — Preprocessing Refactoring (1-2 weeks)

**PR-3: Reader strategy pattern for `DataFormatPreprocessor`** (L)
Collapse 18 subclasses into one `DataFormatPreprocessor` with injected format reader:
```python
class DataFormatReader(ABC):
    @abstractmethod
    def read(self, path: str, **kwargs) -> pd.DataFrame: ...

class CSVReader(DataFormatReader):
    def read(self, path, **kwargs): return pd.read_csv(path, **kwargs)

# One 5-line reader per format instead of one 50-line class per format
```
`preprocessing.py` drops from 2407 → ~1000 lines.
- New package: `ludwig/data/readers/`
- Add unit tests per reader (normal, missing file, malformed, empty)

**PR-4: Base preprocessing module** (M)
Extract shared logic (missing value handling, dtype casting, reshaping) into `BasePreprocessingModule`. Have all feature `_Preprocessing` inner classes inherit from it.
- Files: `features/base_feature.py`, `features/image_feature.py`, `features/audio_feature.py`, `features/category_feature.py`

---

### Phase 3 — Trainer Modularization (1-2 weeks)

**PR-5: Trainer composition over mixin inheritance** (L)
```python
# Before: class Trainer(CheckpointMixin, EarlyStoppingMixin, MetricsMixin, ProfilingMixin, BaseTrainer)
# After:
class Trainer(BaseTrainer):
    def __init__(self, config, backend,
                 checkpointer: CheckpointService,
                 early_stopper: EarlyStoppingService,
                 metrics_collector: MetricsCollectionService,
                 profiler: ProfilingService | None = None): ...
```
- `Trainer.__init__` shrinks from 24 params to 5
- Each service is independently testable
- Files: `trainers/trainer.py`, `trainers/mixins.py` → `trainers/services/`

**PR-6: Decompose `train_loop()`** (M)
Break 240-line method into `_forward_pass()`, `_backward_pass()`, `_update_metrics()`, `_maybe_checkpoint()`, `_maybe_early_stop()` — each <50 lines.
- Files: `trainers/trainer.py`

---

### Phase 4 — Error Handling (1 week)

**PR-7: Fix bare `except Exception:` handlers** (M)
112 instances. Replace with specific exception types + logging. At minimum:
- `check.py:32` — add `logger.exception()`
- `strings_utils.py:138` — replace `return None` with typed exception
- `image_utils.py:98` — replace `return None` with logged warning
- All handlers that discard errors silently in preprocessing/data loading paths

**PR-7b: Fix `collect_activations()` OOM** (M)
Add chunked streaming with disk offload. `collect.py`.

---

### Phase 5 — Structure & Dead Code (1 week)

**PR-8: Split `LudwigModel` (`api.py`)** (XL)
```
api.py (~600 lines) — train(), evaluate(), predict(), save(), load() only
experiment.py    — run_experiment(), hyperopt integration
serve_v2.py      — already partially extracted; complete it
explain.py       — already partially extracted
```

**PR-9: Promote `PassthroughInputFeature`** (S)
Remove `create_passthrough_input_feature()` factory (`base_feature.py:648-703`). Create `features/passthrough_feature.py` with a proper class in the feature registry.

**PR-10: Rename misleading identifiers** (S)
`NoneTrainer` → `InferenceOnlyTrainer`, `LocalPreprocessingMixin` → `LocalDataProcessingMixin`, `CacheManager` → `PreprocessedDataCache`

---

### Phase 6 — Test Coverage (Ongoing)

**PR-11: Unit tests for format readers** (M)
After PR-3: `tests/ludwig/data/readers/test_*.py` — normal read, missing file, malformed, empty.

**PR-12: Unit tests for `collect.py`** (S)
Test with 2-layer model; verify output shapes; verify chunked mode.

**PR-13: Unit tests for `BatchInferModel`** (S)
Test inner class at `backend/ray.py:816`.

**PR-14: Unit tests for RLHF trainers** (M)
`KTOTrainer`, `ORPOTrainer`, `GRPOTrainer` — unit tests with tiny models, no GPU required.

---

### Phase 7 — Documentation & Contributor Experience (1 week)

**PR-15: Feature contributor guide** (S)
`docs/developer_guide/adding_a_feature_type.md` — schema class, feature module, mixin pattern, required vs optional methods, test template.

**PR-16: Resolve all TODOs** (M)
Audit all 172. Fix bugs; convert design questions to GitHub issues; delete the comment.

---

## Summary Table

| PR | Description | Size | Priority |
|----|-------------|------|----------|
| PR-0a | Fix typos + silent bugs | S | P0 |
| PR-0b | Fix TODO bugs | S | P0 |
| PR-7 | Fix bare `except Exception:` | M | P0 |
| PR-7b | Fix `collect_activations()` OOM | M | P0 |
| PR-1 | TypedDict for data contracts | L | P0 |
| PR-3 | Reader strategy for format preprocessors | L | P0 |
| PR-2 | Backend capabilities dataclass | S | P1 |
| PR-4 | Base preprocessing module | M | P1 |
| PR-5 | Trainer composition over inheritance | L | P1 |
| PR-6 | Decompose `train_loop()` | M | P1 |
| PR-11 | Unit tests for format readers | M | P1 |
| PR-12 | Unit tests for `collect.py` | S | P1 |
| PR-13 | Unit tests for `BatchInferModel` | S | P1 |
| PR-14 | Unit tests for RLHF trainers | M | P1 |
| PR-8 | Split `LudwigModel` | XL | P2 |
| PR-9 | Promote `PassthroughInputFeature` | S | P2 |
| PR-10 | Rename misleading identifiers | S | P2 |
| PR-15 | Feature contributor guide | S | P2 |
| PR-16 | Resolve all TODOs | M | P2 |

**Recommended start order:** PR-0a → PR-7 → PR-3 → PR-1 → PR-5
**Total estimated effort:** 6-8 weeks of focused engineering (PRs within phases can be parallelized).
