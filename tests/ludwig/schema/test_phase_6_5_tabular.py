"""Phase 6.5 advanced tabular — schema + preset unit tests."""

from __future__ import annotations

import pytest

from ludwig.error import ConfigValidationError
from ludwig.presets import apply_preset, QUALITY_PRESETS


class TestRealMLPPreset:
    def test_preset_registered(self):
        assert "tabular_realmlp" in QUALITY_PRESETS

    def test_preset_sets_number_robust_scaling(self):
        preset = QUALITY_PRESETS["tabular_realmlp"]
        # 'iq' is Ludwig's interquartile-range normalizer, the closest match to the
        # RobustScaler used by the RealMLP paper.
        assert preset["defaults"]["number"]["preprocessing"]["normalization"] == "iq"

    def test_preset_uses_adamw_cosine(self):
        trainer = QUALITY_PRESETS["tabular_realmlp"]["trainer"]
        assert trainer["optimizer"]["type"] == "adamw"
        assert trainer["learning_rate_scheduler"]["decay"] == "cosine"

    def test_user_config_overrides_preset(self):
        """User config wins on any collision."""
        user_cfg = {
            "input_features": [{"name": "x", "type": "number"}],
            "output_features": [{"name": "y", "type": "binary"}],
            "trainer": {"epochs": 5},
        }
        merged = apply_preset(user_cfg, "tabular_realmlp")
        # User's 5 wins over preset's 300.
        assert merged["trainer"]["epochs"] == 5
        # But the number-feature normalization is still set from the preset.
        assert merged["defaults"]["number"]["preprocessing"]["normalization"] == "iq"


class TestRealMLPSchemaEnum:
    """ECDModelConfig schema must advertise the new preset name."""

    def test_preset_accepted_on_ecd(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "model_type": "ecd",
                "preset": "tabular_realmlp",
                "input_features": [{"name": "x", "type": "number"}],
                "output_features": [{"name": "y", "type": "binary"}],
            }
        )
        assert cfg.preset == "tabular_realmlp"

    def test_unknown_preset_rejected(self):
        from ludwig.schema.model_config import ModelConfig

        # Preset application runs before schema validation, so an unknown name surfaces as a
        # plain ValueError from ludwig.presets.apply_preset; the schema enum would otherwise
        # catch it as ConfigValidationError. Accept either — both are hard rejections.
        with pytest.raises((ConfigValidationError, ValueError)):
            ModelConfig.from_dict(
                {
                    "model_type": "ecd",
                    "preset": "i_made_this_up",
                    "input_features": [{"name": "x", "type": "number"}],
                    "output_features": [{"name": "y", "type": "binary"}],
                }
            )


class TestTabPFNV2CombinerSchema:
    """Schema fields for the TabPFN v2 combiner — no `tabpfn` package required here."""

    def test_schema_registers_and_parses(self):
        from ludwig.schema.combiners.tabpfn_v2 import TabPFNV2CombinerConfig
        from ludwig.schema.combiners.utils import combiner_config_registry

        assert "tabpfn_v2" in combiner_config_registry
        cfg = TabPFNV2CombinerConfig.model_validate(
            {
                "type": "tabpfn_v2",
                "output_size": 256,
                "n_estimators": 8,
                "device": "cuda",
            }
        )
        assert cfg.type == "tabpfn_v2"
        assert cfg.output_size == 256
        assert cfg.n_estimators == 8
        assert cfg.device == "cuda"

    def test_defaults(self):
        from ludwig.schema.combiners.tabpfn_v2 import TabPFNV2CombinerConfig

        cfg = TabPFNV2CombinerConfig.model_validate({"type": "tabpfn_v2"})
        assert cfg.output_size == 128
        assert cfg.tabpfn_hidden_size == 512
        assert cfg.n_estimators == 4
        assert cfg.device == "auto"

    def test_device_enum_rejects_unknown(self):
        from ludwig.schema.combiners.tabpfn_v2 import TabPFNV2CombinerConfig

        with pytest.raises(Exception):  # pydantic ValidationError
            TabPFNV2CombinerConfig.model_validate({"type": "tabpfn_v2", "device": "tpu"})

    def test_full_ecd_config_with_tabpfn_v2(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "model_type": "ecd",
                "input_features": [
                    {"name": "a", "type": "number"},
                    {"name": "b", "type": "number"},
                ],
                "output_features": [{"name": "y", "type": "binary"}],
                "combiner": {"type": "tabpfn_v2", "output_size": 64, "n_estimators": 2},
            }
        )
        assert cfg.combiner.type == "tabpfn_v2"
        assert cfg.combiner.output_size == 64
        assert cfg.combiner.n_estimators == 2


class TestTabPFNV2CombinerInit:
    """The combiner class raises a clear error when the optional ``tabpfn`` package is missing.

    When the dep is installed the import succeeds and instantiation wires the projection head.
    """

    def test_import_error_message_cites_pip_install(self):
        try:
            import tabpfn  # noqa: F401
        except ImportError:
            from ludwig.combiners.tabpfn_v2_combiner import TabPFNV2Combiner
            from ludwig.schema.combiners.tabpfn_v2 import TabPFNV2CombinerConfig

            with pytest.raises(ImportError, match="pip install tabpfn"):
                TabPFNV2Combiner(config=TabPFNV2CombinerConfig())
        else:
            pytest.skip("tabpfn is installed; skipping missing-package error-message test")
