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
