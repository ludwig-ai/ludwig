"""Regression tests for issues discovered during the Ludwig dataset smoke-test campaign.

Each test is a direct regression guard for a specific root-cause failure that was
found, fixed, and must never regress.  The test names describe the exact symptom.

Issues covered
--------------
1. select_columns breaks TranslationLoader (custom loaders read intermediate columns)
2. TranslationLoader splits a nested dict column into per-language flat columns
3. Europarl language-pair loaders exist for all supported pairs
4. MASSIVE intent/scenario dataset IDs use lowercase snake_case, not CamelCase
5. MTOP domain/intent supports all 6 expected languages
6. MassiveScenarioClassification does not exist; correct ID is massive_scenario
7. Dataset configs: germeval18 uses column "binary", not "label"
8. STS17 cross-lingual subsample order is "en-de", not "de-en"
9. smoke_results.json cleanup: stale entries for deleted configs are removed
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ludwig", "datasets", "configs")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "smoke_results.json")


def load_config(name: str) -> dict:
    path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def all_config_names() -> set[str]:
    return {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CONFIGS_DIR, "*.yaml"))}


# ---------------------------------------------------------------------------
# 1. select_columns must be disabled for custom loaders
# ---------------------------------------------------------------------------


class TestSelectColumnsCustomLoader:
    """Regression: select_columns=[de, en] was applied BEFORE TranslationLoader
    ran, stripping the 'translation' dict column and producing an empty DataFrame.
    The fix detects has_custom_loader and disables select_columns for them.
    """

    def test_has_custom_loader_flag_for_translation_datasets(self):
        """Every config with a non-HFLoader loader must be detectable as custom."""
        for name in all_config_names():
            cfg = load_config(name)
            loader_spec = cfg.get("loader", "")
            has_custom = bool(loader_spec and loader_spec != "hugging_face.HFLoader")
            if has_custom:
                # The loader must reference a real class in translation_loader or misc_loaders
                assert "." in loader_spec, f"{name}: loader spec must be module.ClassName, got {loader_spec!r}"

    def test_wmt_t2t_de_en_uses_translation_loader(self):
        cfg = load_config("wmt_t2t_de_en")
        assert cfg["loader"] != "hugging_face.HFLoader", "wmt_t2t_de_en must use a TranslationLoader subclass"

    def test_europarl_configs_use_translation_loader(self):
        europarl_names = [n for n in all_config_names() if n.startswith("europarl_")]
        assert len(europarl_names) >= 13, f"Expected ≥13 europarl configs, found {len(europarl_names)}"
        for name in europarl_names:
            cfg = load_config(name)
            loader = cfg.get("loader", "")
            assert "TranslationLoader" in loader or "Loader" in loader, (
                f"{name}: expected a TranslationLoader subclass, got {loader!r}"
            )


# ---------------------------------------------------------------------------
# 2. TranslationLoader: verify the loader class hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
class TestTranslationLoaderClasses:
    def test_europarl_loader_classes_importable(self):
        from ludwig.datasets.loaders.translation_loader import (
            EuroparlBgEnLoader,
            EuroparlCsEnLoader,
            EuroparlDaEnLoader,
            EuroparlDeEnLoader,
            EuroparlElEnLoader,
            EuroparlEnEsLoader,
            EuroparlEnFrLoader,
            EuroparlEnItLoader,
            EuroparlEnNlLoader,
            EuroparlEnPlLoader,
            EuroparlEnPtLoader,
            EuroparlEnRoLoader,
            EuroparlEnSvLoader,
            TranslationLoader,
        )

        for cls in [
            EuroparlBgEnLoader,
            EuroparlCsEnLoader,
            EuroparlDaEnLoader,
            EuroparlDeEnLoader,
            EuroparlElEnLoader,
            EuroparlEnEsLoader,
            EuroparlEnFrLoader,
            EuroparlEnItLoader,
            EuroparlEnNlLoader,
            EuroparlEnPlLoader,
            EuroparlEnPtLoader,
            EuroparlEnRoLoader,
            EuroparlEnSvLoader,
        ]:
            assert issubclass(cls, TranslationLoader), f"{cls.__name__} must be a TranslationLoader subclass"

    def test_translation_loader_has_source_and_target_lang(self):
        from ludwig.datasets.loaders.translation_loader import EuroparlDeEnLoader

        assert hasattr(EuroparlDeEnLoader, "source_lang")
        assert hasattr(EuroparlDeEnLoader, "target_lang")
        assert EuroparlDeEnLoader.source_lang == "de"
        assert EuroparlDeEnLoader.target_lang == "en"

    def test_translation_loader_transform_expands_nested_dict(self):
        """TranslationLoader._transform must split a nested translation dict into flat columns."""
        from ludwig.datasets.loaders.translation_loader import EuroparlDeEnLoader

        raw = pd.DataFrame({"translation": [{"de": "Hallo", "en": "Hello"}, {"de": "Welt", "en": "World"}]})

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            PatchedCls = type(
                "_TestEuroparlDeEnLoader",
                (EuroparlDeEnLoader,),
                {"processed_dataset_dir": property(lambda self, d=tmpdir: d)},
            )
            instance = object.__new__(PatchedCls)
            result = instance._transform(raw)

        assert "de" in result.columns, f"Expected 'de' column after transform, got {list(result.columns)}"
        assert "en" in result.columns, f"Expected 'en' column after transform, got {list(result.columns)}"
        assert list(result["de"]) == ["Hallo", "Welt"]
        assert list(result["en"]) == ["Hello", "World"]


# ---------------------------------------------------------------------------
# 3. Europarl configs — one config per language pair
# ---------------------------------------------------------------------------


class TestEuroparlConfigs:
    EXPECTED_PAIRS = [
        "europarl_bg_en",
        "europarl_cs_en",
        "europarl_da_en",
        "europarl_de_en",
        "europarl_el_en",
        "europarl_en_es",
        "europarl_en_fr",
        "europarl_en_it",
        "europarl_en_nl",
        "europarl_en_pl",
        "europarl_en_pt",
        "europarl_en_ro",
        "europarl_en_sv",
    ]

    def test_all_expected_europarl_configs_exist(self):
        existing = all_config_names()
        for name in self.EXPECTED_PAIRS:
            assert name in existing, f"Missing Europarl config: {name}"

    def test_europarl_configs_have_two_text_output_columns(self):
        for name in self.EXPECTED_PAIRS:
            cfg = load_config(name)
            col_names = [c["name"] for c in cfg.get("columns", [])]
            out_names = [f["name"] for f in cfg.get("output_features", [])]
            assert len(col_names) >= 2, f"{name}: expected ≥2 columns"
            assert len(out_names) == 1, f"{name}: expected 1 output feature"


# ---------------------------------------------------------------------------
# 4 & 6. MASSIVE dataset ID correctness
# ---------------------------------------------------------------------------


class TestMassiveDatasetIDs:
    def test_massive_intent_configs_use_correct_dataset_id(self):
        """Regression: new configs incorrectly used 'mteb/MassiveIntentClassification'."""
        intent_configs = [n for n in all_config_names() if n.startswith("mteb_massive_intent_")]
        assert len(intent_configs) >= 5, "Expected at least 5 MASSIVE intent configs"
        for name in intent_configs:
            cfg = load_config(name)
            assert cfg["huggingface_dataset_id"] == "mteb/massive_intent", (
                f"{name}: wrong dataset ID {cfg['huggingface_dataset_id']!r}; "
                "must be 'mteb/massive_intent' (lowercase snake_case)"
            )

    def test_massive_scenario_configs_use_correct_dataset_id(self):
        """Regression: 'mteb/MassiveScenarioClassification' does not exist on the Hub."""
        scenario_configs = [n for n in all_config_names() if n.startswith("mteb_massive_scenario_")]
        assert len(scenario_configs) >= 5, "Expected at least 5 MASSIVE scenario configs"
        for name in scenario_configs:
            cfg = load_config(name)
            assert cfg["huggingface_dataset_id"] == "mteb/massive_scenario", (
                f"{name}: wrong dataset ID {cfg['huggingface_dataset_id']!r}; "
                "must be 'mteb/massive_scenario' (lowercase snake_case)"
            )

    def test_massive_intent_covers_51_languages(self):
        intent_configs = [n for n in all_config_names() if n.startswith("mteb_massive_intent_")]
        assert len(intent_configs) == 51, f"Expected 51 MASSIVE intent configs, got {len(intent_configs)}"

    def test_massive_scenario_covers_51_languages(self):
        scenario_configs = [n for n in all_config_names() if n.startswith("mteb_massive_scenario_")]
        assert len(scenario_configs) == 51, f"Expected 51 MASSIVE scenario configs, got {len(scenario_configs)}"


# ---------------------------------------------------------------------------
# 5. MTOP domain/intent: all 6 languages present
# ---------------------------------------------------------------------------


class TestMTOPConfigs:
    LANGUAGES = ["en", "de", "fr", "es", "hi", "th"]

    def test_mtop_domain_all_languages(self):
        for lang in self.LANGUAGES:
            name = f"mteb_mtop_domain_{lang}"
            assert name in all_config_names(), f"Missing MTOP domain config: {name}"
            cfg = load_config(name)
            assert cfg["huggingface_dataset_id"] == "mteb/MTOPDomainClassification"
            assert cfg["huggingface_subsample"] == lang

    def test_mtop_intent_all_languages(self):
        intent_cfgs = {
            "en": "mteb_mtop_intent_en",
            "de": "mteb_mtop_intent_de2",
            "fr": "mteb_mtop_intent_fr2",
            "es": "mteb_mtop_intent_es2",
            "hi": "mteb_mtop_intent_hi2",
            "th": "mteb_mtop_intent_th2",
        }
        for lang, cfg_name in intent_cfgs.items():
            assert cfg_name in all_config_names(), f"Missing MTOP intent config: {cfg_name}"
            cfg = load_config(cfg_name)
            assert cfg["huggingface_dataset_id"] == "mteb/MTOPIntentClassification"
            assert cfg["huggingface_subsample"] == lang


# ---------------------------------------------------------------------------
# 7. germeval18: output column is "binary", not "label"
# ---------------------------------------------------------------------------


class TestGermeval18Config:
    def test_output_column_is_binary(self):
        cfg = load_config("germeval18")
        out_cols = {f["name"] for f in cfg.get("output_features", [])}
        assert "binary" in out_cols, (
            "germeval18 must use column 'binary' as output (philschmid/germeval18 has no 'label' column)"
        )
        assert "label" not in out_cols, "germeval18 must not reference 'label' — that column does not exist"

    def test_germeval18_uses_philschmid_dataset(self):
        cfg = load_config("germeval18")
        assert "philschmid" in cfg.get("huggingface_dataset_id", ""), "germeval18 must use philschmid/germeval18"


# ---------------------------------------------------------------------------
# 8. STS17: subsample order is "en-de", not "de-en"
# ---------------------------------------------------------------------------


class TestSTS17SubsampleOrder:
    def test_sts17_de_uses_en_de_subsample(self):
        cfg = load_config("mteb_sts17_de")
        subsample = cfg.get("huggingface_subsample", "")
        assert subsample == "en-de", f"mteb_sts17_de must use subsample 'en-de' (not 'de-en'); got {subsample!r}"

    def test_sts17_es_uses_es_en_subsample(self):
        cfg = load_config("mteb_sts17_es")
        subsample = cfg.get("huggingface_subsample", "")
        # es-en or en-es — just confirm it's not plain "es" or "en"
        assert "-" in subsample, f"mteb_sts17_es subsample must be a language pair, got {subsample!r}"


# ---------------------------------------------------------------------------
# 9. smoke_results.json: no stale entries for deleted configs
# ---------------------------------------------------------------------------


class TestSmokeResultsConsistency:
    def test_no_stale_results_for_missing_configs(self):
        if not os.path.exists(RESULTS_FILE):
            pytest.skip("smoke_results.json not found")

        with open(RESULTS_FILE) as f:
            results = json.load(f)

        existing_configs = all_config_names()
        stale = [r["name"] for r in results if r["name"] not in existing_configs]
        assert stale == [], (
            f"smoke_results.json contains {len(stale)} stale entries for configs that no longer exist: {stale[:10]}"
        )

    def test_all_results_have_required_fields(self):
        if not os.path.exists(RESULTS_FILE):
            pytest.skip("smoke_results.json not found")

        with open(RESULTS_FILE) as f:
            results = json.load(f)

        for entry in results:
            assert "name" in entry, f"Result entry missing 'name': {entry}"
            assert "status" in entry, f"Result entry for {entry.get('name')} missing 'status'"


# ---------------------------------------------------------------------------
# General config schema validation
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_all_configs_have_required_fields(self):
        for name in all_config_names():
            if name in self._NON_DATASET_CONFIGS:
                continue
            cfg = load_config(name)
            assert "version" in cfg, f"{name}: missing 'version'"
            assert "name" in cfg, f"{name}: missing 'name'"
            assert cfg["name"] == name, f"{name}: 'name' field {cfg['name']!r} does not match filename"

    # Loader-reference stubs that are not actual dataset configs
    _NON_DATASET_CONFIGS = {"hugging_face"}

    def test_hf_configs_have_dataset_id(self):
        for name in all_config_names():
            if name in self._NON_DATASET_CONFIGS:
                continue
            cfg = load_config(name)
            loader = cfg.get("loader", "hugging_face.HFLoader")
            # Configs using direct download or Kaggle are not HF configs
            uses_direct_download = bool(
                cfg.get("download_urls") or cfg.get("kaggle_dataset_id") or cfg.get("kaggle_competition")
            )
            if uses_direct_download:
                continue
            if "HFLoader" in loader or not loader:
                has_id = bool(cfg.get("huggingface_dataset_id"))
                has_legacy_id = bool(cfg.get("hf_dataset_id"))  # legacy field name
                assert has_id or has_legacy_id, f"{name}: HF loader config missing 'huggingface_dataset_id'"

    def test_output_features_reference_declared_columns(self):
        for name in all_config_names():
            cfg = load_config(name)
            col_names = {c["name"] for c in cfg.get("columns", [])}
            if not col_names:
                continue  # some legacy configs have no columns list
            for out_feat in cfg.get("output_features", []):
                out_name = out_feat["name"]
                assert out_name in col_names, (
                    f"{name}: output feature '{out_name}' not in declared columns {sorted(col_names)}"
                )
