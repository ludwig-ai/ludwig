"""Tests for expanded PEFT adapter support."""

import pytest

from ludwig.schema.llms.peft import adapter_registry


class TestAdapterRegistry:
    def test_all_adapters_registered(self):
        expected = {"lora", "adalora", "ia3", "vera", "loha", "lokr", "fourierft", "boft"}
        assert expected.issubset(set(adapter_registry.keys()))

    @pytest.mark.parametrize("adapter_type", ["lora", "adalora", "ia3", "vera", "loha", "lokr", "fourierft", "boft"])
    def test_adapter_creates_valid_peft_config(self, adapter_type):
        cls = adapter_registry[adapter_type]
        inst = cls.model_validate({"type": adapter_type})
        peft_config = inst.to_config()
        assert peft_config is not None

    @pytest.mark.parametrize("adapter_type", ["vera", "loha", "lokr", "fourierft", "boft"])
    def test_new_adapter_has_target_modules(self, adapter_type):
        cls = adapter_registry[adapter_type]
        inst = cls.model_validate({"type": adapter_type})
        assert hasattr(inst, "target_modules")


class TestLoraPlus:
    def test_loraplus_lr_ratio_default_none(self):
        cls = adapter_registry["lora"]
        inst = cls.model_validate({"type": "lora"})
        assert inst.loraplus_lr_ratio is None

    def test_loraplus_lr_ratio_set(self):
        cls = adapter_registry["lora"]
        inst = cls.model_validate({"type": "lora", "loraplus_lr_ratio": 8.0})
        assert inst.loraplus_lr_ratio == 8.0

    def test_loraplus_param_groups(self):
        import torch.nn as nn

        from ludwig.modules.optimization_modules import _create_loraplus_param_groups

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(10, 10)
                self.lora_A = nn.Linear(10, 4, bias=False)
                self.lora_B = nn.Linear(4, 10, bias=False)

        model = MockModel()
        groups = _create_loraplus_param_groups(model, {"lr": 0.001}, loraplus_lr_ratio=8.0)

        assert len(groups) == 3
        lrs = sorted(g["lr"] for g in groups)
        assert lrs == [0.001, 0.001, 0.008]


class TestECDEncoderAdapter:
    def test_adapter_field_exists(self):
        from ludwig.schema.encoders.base import DenseEncoderConfig

        inst = DenseEncoderConfig.model_validate({"type": "dense"})
        assert inst.adapter is None

    def test_adapter_field_accepts_dict(self):
        from ludwig.schema.encoders.base import DenseEncoderConfig

        inst = DenseEncoderConfig.model_validate({"type": "dense", "adapter": {"type": "lora", "r": 8}})
        assert inst.adapter == {"type": "lora", "r": 8}


class TestNamedAdapters:
    """Unit tests for the multi-adapter schema (`adapters:` plural, mutually exclusive with `adapter:`)."""

    def _llm_base(self):
        return {
            "model_type": "llm",
            "base_model": "sshleifer/tiny-gpt2",
            "input_features": [{"name": "p", "type": "text"}],
            "output_features": [{"name": "r", "type": "text"}],
            "trainer": {"type": "finetune"},
        }

    def test_named_adapters_config_validates(self):
        from ludwig.schema.llms.peft import NamedAdaptersConfig

        cfg = NamedAdaptersConfig.model_validate(
            {
                "adapters": {"a": {"type": "lora", "r": 8}, "b": {"type": "lora", "r": 16}},
                "active": "a",
            }
        )
        assert cfg.active == "a"
        assert list(cfg.adapters.keys()) == ["a", "b"]
        assert cfg.merge is None

    def test_merge_config_validates(self):
        from ludwig.schema.llms.peft import NamedAdaptersConfig

        cfg = NamedAdaptersConfig.model_validate(
            {
                "adapters": {"a": {"type": "lora"}, "b": {"type": "lora"}},
                "merge": {
                    "name": "m",
                    "sources": ["a", "b"],
                    "weights": [0.7, 0.3],
                    "combination_type": "ties",
                    "density": 0.3,
                },
            }
        )
        assert cfg.merge.name == "m"
        assert cfg.merge.combination_type == "ties"
        assert cfg.merge.density == 0.3

    def test_llm_config_accepts_plural(self):
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {**self._llm_base(), "adapters": {"adapters": {"a": {"type": "lora"}}}}
        model_cfg = ModelConfig.from_dict(cfg)
        assert model_cfg.adapter is None
        assert model_cfg.adapters is not None
        assert "a" in model_cfg.adapters.adapters

    def test_llm_config_accepts_singular(self):
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {**self._llm_base(), "adapter": {"type": "lora", "r": 8}}
        model_cfg = ModelConfig.from_dict(cfg)
        assert model_cfg.adapter is not None
        assert model_cfg.adapters is None

    def test_both_adapter_and_adapters_rejected(self):
        from ludwig.error import ConfigValidationError
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {
            **self._llm_base(),
            "adapter": {"type": "lora", "r": 8},
            "adapters": {"adapters": {"a": {"type": "lora"}}},
        }
        with pytest.raises(ConfigValidationError, match="both `adapter:` and `adapters:`"):
            ModelConfig.from_dict(cfg)

    def test_empty_adapters_rejected(self):
        from ludwig.error import ConfigValidationError
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {**self._llm_base(), "adapters": {"adapters": {}}}
        with pytest.raises(ConfigValidationError, match="at least one entry"):
            ModelConfig.from_dict(cfg)

    def test_active_must_reference_known_adapter(self):
        from ludwig.error import ConfigValidationError
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {
            **self._llm_base(),
            "adapters": {"adapters": {"a": {"type": "lora"}}, "active": "b"},
        }
        with pytest.raises(ConfigValidationError, match="does not match any"):
            ModelConfig.from_dict(cfg)

    def test_active_may_point_at_merged_adapter(self):
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {
            **self._llm_base(),
            "adapters": {
                "adapters": {"a": {"type": "lora"}, "b": {"type": "lora"}},
                "active": "m",
                "merge": {"name": "m", "sources": ["a", "b"], "combination_type": "linear"},
            },
        }
        model_cfg = ModelConfig.from_dict(cfg)
        assert model_cfg.adapters.active == "m"

    def test_merge_sources_must_exist(self):
        from ludwig.error import ConfigValidationError
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {
            **self._llm_base(),
            "adapters": {
                "adapters": {"a": {"type": "lora"}},
                "merge": {"name": "m", "sources": ["a", "ghost"]},
            },
        }
        with pytest.raises(ConfigValidationError, match="unknown adapter names"):
            ModelConfig.from_dict(cfg)

    def test_merge_weights_length_must_match_sources(self):
        from ludwig.error import ConfigValidationError
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {
            **self._llm_base(),
            "adapters": {
                "adapters": {"a": {"type": "lora"}, "b": {"type": "lora"}},
                "merge": {"name": "m", "sources": ["a", "b"], "weights": [0.5]},
            },
        }
        with pytest.raises(ConfigValidationError, match="Lengths must match"):
            ModelConfig.from_dict(cfg)

    def test_merge_name_cannot_collide_with_source(self):
        from ludwig.error import ConfigValidationError
        from ludwig.schema.model_types.base import ModelConfig

        cfg = {
            **self._llm_base(),
            "adapters": {
                "adapters": {"a": {"type": "lora"}, "b": {"type": "lora"}},
                "merge": {"name": "a", "sources": ["a", "b"]},
            },
        }
        with pytest.raises(ConfigValidationError, match="collides with an existing source"):
            ModelConfig.from_dict(cfg)


class TestInitializeAdapterMulti:
    """Unit tests for `_initialize_multi_adapters` (no base model download)."""

    def test_materialize_adapter_config_from_dict(self):
        from ludwig.utils.llm_utils import _materialize_adapter_config

        cfg = _materialize_adapter_config({"type": "lora", "r": 8})
        assert hasattr(cfg, "to_config")
        assert cfg.type == "lora"
        assert cfg.r == 8

    def test_materialize_adapter_config_unknown_type_raises(self):
        from ludwig.utils.llm_utils import _materialize_adapter_config

        with pytest.raises(ValueError, match="Unknown adapter type"):
            _materialize_adapter_config({"type": "definitely-not-a-real-adapter"})

    def test_materialize_adapter_config_missing_type_raises(self):
        from ludwig.utils.llm_utils import _materialize_adapter_config

        with pytest.raises(ValueError, match="missing required `type`"):
            _materialize_adapter_config({"r": 8})
