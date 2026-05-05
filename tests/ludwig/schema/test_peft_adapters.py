"""Tests for expanded PEFT adapter support."""

import pytest

from ludwig.schema.llms.peft import adapter_registry

_ALL_ADAPTERS = [
    "lora",
    "adalora",
    "ia3",
    "vera",
    "loha",
    "lokr",
    "fourierft",
    "boft",
    "tinylora",
    "c3a",
    "oft",
    "hra",
    "waveft",
    "ln_tuning",
    "vblora",
]
_ADAPTERS_WITH_TARGET_MODULES = [
    "vera",
    "loha",
    "lokr",
    "fourierft",
    "boft",
    "tinylora",
    "c3a",
    "oft",
    "hra",
    "waveft",
    "ln_tuning",
    "vblora",
]


class TestAdapterRegistry:
    def test_all_adapters_registered(self):
        expected = set(_ALL_ADAPTERS)
        assert expected.issubset(set(adapter_registry.keys()))

    @pytest.mark.parametrize("adapter_type", _ALL_ADAPTERS)
    def test_adapter_creates_valid_peft_config(self, adapter_type):
        cls = adapter_registry[adapter_type]
        inst = cls.model_validate({"type": adapter_type})
        peft_config = inst.to_config()
        assert peft_config is not None

    @pytest.mark.parametrize("adapter_type", _ADAPTERS_WITH_TARGET_MODULES)
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


class TestLoraInitializers:
    """Tests for PiSSA, EVA, CorDA, LoftQ, and other init_lora_weights options."""

    @pytest.mark.parametrize("init", ["default", "gaussian", "pissa", "olora", "orthogonal"])
    def test_init_lora_weights_string_options(self, init):
        from ludwig.schema.llms.peft import LoraConfig

        cfg = LoraConfig.model_validate({"type": "lora", "init_lora_weights": init})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        expected = True if init == "default" else init
        assert peft_cfg.init_lora_weights == expected

    def test_pissa_init(self):
        from ludwig.schema.llms.peft import LoraConfig

        cfg = LoraConfig.model_validate({"type": "lora", "r": 4, "init_lora_weights": "pissa"})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.init_lora_weights == "pissa"
        assert peft_cfg.r == 4

    def test_eva_init_requires_eva_config(self):
        from pydantic import ValidationError

        from ludwig.schema.llms.peft import LoraConfig

        with pytest.raises(ValidationError, match="eva_config"):
            LoraConfig.model_validate({"type": "lora", "init_lora_weights": "eva"})

    def test_eva_init_with_config(self):
        from ludwig.schema.llms.peft import LoraConfig

        cfg = LoraConfig.model_validate(
            {
                "type": "lora",
                "init_lora_weights": "eva",
                "eva_config": {"rho": 3.0, "tau": 0.95},
            }
        )
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.init_lora_weights == "eva"
        assert peft_cfg.eva_config is not None
        assert peft_cfg.eva_config.rho == 3.0

    def test_loftq_init_requires_loftq_config(self):
        from pydantic import ValidationError

        from ludwig.schema.llms.peft import LoraConfig

        with pytest.raises(ValidationError, match="loftq_config"):
            LoraConfig.model_validate({"type": "lora", "init_lora_weights": "loftq"})

    def test_loftq_init_with_config(self):
        from ludwig.schema.llms.peft import LoraConfig

        cfg = LoraConfig.model_validate(
            {
                "type": "lora",
                "init_lora_weights": "loftq",
                "loftq_config": {"loftq_bits": 4, "loftq_iter": 2},
            }
        )
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.init_lora_weights == "loftq"
        assert peft_cfg.loftq_config["loftq_bits"] == 4
        assert peft_cfg.loftq_config["loftq_iter"] == 2

    def test_corda_init(self):
        from ludwig.schema.llms.peft import LoraConfig

        cfg = LoraConfig.model_validate({"type": "lora", "init_lora_weights": "corda"})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.init_lora_weights == "corda"

    def test_rank_pattern(self):
        from ludwig.schema.llms.peft import LoraConfig

        pattern = {"model.layers.0.self_attn.q_proj": 4, "model.layers.0.self_attn.v_proj": 2}
        cfg = LoraConfig.model_validate({"type": "lora", "r": 8, "rank_pattern": pattern})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.rank_pattern == pattern

    def test_alpha_pattern(self):
        from ludwig.schema.llms.peft import LoraConfig

        pattern = {"model.layers.0.self_attn.q_proj": 16.0}
        cfg = LoraConfig.model_validate({"type": "lora", "alpha_pattern": pattern})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.alpha_pattern == pattern

    def test_layer_replication(self):
        from ludwig.schema.llms.peft import LoraConfig

        cfg = LoraConfig.model_validate({"type": "lora", "layer_replication": [[0, 4], [2, 5]]})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.layer_replication == [(0, 4), (2, 5)]

    def test_default_rank_pattern_is_empty(self):
        from ludwig.schema.llms.peft import LoraConfig

        cfg = LoraConfig.model_validate({"type": "lora"})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.rank_pattern == {}
        assert peft_cfg.alpha_pattern == {}


class TestTinyLoraAdapter:
    def test_defaults(self):
        from ludwig.schema.llms.peft import TinyLoraAdapterConfig

        cfg = TinyLoraAdapterConfig.model_validate({"type": "tinylora"})
        assert cfg.r == 2
        assert cfg.u == 64
        assert cfg.weight_tying == 0.0

    def test_custom_params(self):
        from ludwig.schema.llms.peft import TinyLoraAdapterConfig

        cfg = TinyLoraAdapterConfig.model_validate({"type": "tinylora", "r": 4, "u": 16, "weight_tying": 0.5})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.r == 4
        assert peft_cfg.u == 16
        assert peft_cfg.weight_tying == 0.5

    def test_name_and_description(self):
        from ludwig.schema.llms.peft import TinyLoraAdapterConfig

        assert TinyLoraAdapterConfig.name() == "TinyLoRA"
        assert "SVD" in TinyLoraAdapterConfig.description()


class TestC3AAdapter:
    def test_defaults(self):
        from ludwig.schema.llms.peft import C3AAdapterConfig

        cfg = C3AAdapterConfig.model_validate({"type": "c3a"})
        assert cfg.block_size == 256

    def test_custom_block_size(self):
        from ludwig.schema.llms.peft import C3AAdapterConfig

        cfg = C3AAdapterConfig.model_validate({"type": "c3a", "block_size": 128})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.block_size == 128

    def test_name(self):
        from ludwig.schema.llms.peft import C3AAdapterConfig

        assert C3AAdapterConfig.name() == "C3A"


class TestOFTAdapter:
    def test_defaults(self):
        from ludwig.schema.llms.peft import OFTAdapterConfig

        cfg = OFTAdapterConfig.model_validate({"type": "oft"})
        assert cfg.oft_block_size == 32
        assert not cfg.coft

    def test_coft_enabled(self):
        from ludwig.schema.llms.peft import OFTAdapterConfig

        cfg = OFTAdapterConfig.model_validate({"type": "oft", "coft": True, "eps": 1e-4})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.coft is True
        assert peft_cfg.eps == 1e-4

    def test_name(self):
        from ludwig.schema.llms.peft import OFTAdapterConfig

        assert OFTAdapterConfig.name() == "OFT"


class TestHRAAdapter:
    def test_defaults(self):
        from ludwig.schema.llms.peft import HRAAdapterConfig

        cfg = HRAAdapterConfig.model_validate({"type": "hra"})
        assert cfg.r == 8
        assert not cfg.apply_GS

    def test_gram_schmidt(self):
        from ludwig.schema.llms.peft import HRAAdapterConfig

        cfg = HRAAdapterConfig.model_validate({"type": "hra", "r": 16, "apply_GS": True})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.r == 16
        assert peft_cfg.apply_GS is True

    def test_name(self):
        from ludwig.schema.llms.peft import HRAAdapterConfig

        assert HRAAdapterConfig.name() == "HRA"


class TestWaveFTAdapter:
    def test_defaults(self):
        from ludwig.schema.llms.peft import WaveFTAdapterConfig

        cfg = WaveFTAdapterConfig.model_validate({"type": "waveft"})
        assert cfg.wavelet_family == "db1"
        assert cfg.n_frequency == 2592

    def test_custom_wavelet(self):
        from ludwig.schema.llms.peft import WaveFTAdapterConfig

        cfg = WaveFTAdapterConfig.model_validate({"type": "waveft", "wavelet_family": "db2", "n_frequency": 512})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.wavelet_family == "db2"
        assert peft_cfg.n_frequency == 512

    def test_name(self):
        from ludwig.schema.llms.peft import WaveFTAdapterConfig

        assert WaveFTAdapterConfig.name() == "WaveFT"


class TestLNTuningAdapter:
    def test_defaults(self):
        from ludwig.schema.llms.peft import LNTuningAdapterConfig

        cfg = LNTuningAdapterConfig.model_validate({"type": "ln_tuning"})
        assert cfg.target_modules is None

    def test_custom_target_modules(self):
        from ludwig.schema.llms.peft import LNTuningAdapterConfig

        cfg = LNTuningAdapterConfig.model_validate({"type": "ln_tuning", "target_modules": ["norm1", "norm2"]})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.target_modules == ["norm1", "norm2"]

    def test_name(self):
        from ludwig.schema.llms.peft import LNTuningAdapterConfig

        assert LNTuningAdapterConfig.name() == "LN-Tuning"


class TestVBLoRAAdapter:
    def test_defaults(self):
        from ludwig.schema.llms.peft import VBLoRAAdapterConfig

        cfg = VBLoRAAdapterConfig.model_validate({"type": "vblora"})
        assert cfg.r == 4
        assert cfg.num_vectors == 256
        assert cfg.topk == 2

    def test_custom_params(self):
        from ludwig.schema.llms.peft import VBLoRAAdapterConfig

        cfg = VBLoRAAdapterConfig.model_validate({"type": "vblora", "r": 8, "num_vectors": 128, "topk": 4})
        peft_cfg = cfg.to_config(task_type="CAUSAL_LM")
        assert peft_cfg.r == 8
        assert peft_cfg.num_vectors == 128
        assert peft_cfg.topk == 4

    def test_name(self):
        from ludwig.schema.llms.peft import VBLoRAAdapterConfig

        assert VBLoRAAdapterConfig.name() == "VBLoRA"


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
