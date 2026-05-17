"""Unit tests for DPO-family trainers (DPOTrainer, KTOTrainer, ORPOTrainer, GRPOTrainer).

These tests verify trainer construction and attribute initialization without a GPU
or a real LLM model. We bypass Trainer.__init__ via patch and inject a mock config.
"""

from unittest.mock import MagicMock, patch

from ludwig.trainers.registry import get_llm_trainers_registry
from ludwig.trainers.trainer import Trainer
from ludwig.trainers.trainer_dpo import DPOTrainer, GRPOTrainer, KTOTrainer, ORPOTrainer


def _noop_trainer_init(self, *args, **kwargs):
    """Replaces Trainer.__init__ so subclass __init__ can run without full model/config setup."""


def _make_dpo(cls, extra_config_attrs: dict | None = None):
    """Create a DPO-family trainer instance, bypassing Trainer.__init__."""
    trainer = cls.__new__(cls)
    trainer.config = MagicMock(spec=[])
    # DPOTrainer reads these attributes from config
    trainer.config.dpo_beta = 0.1
    trainer.config.dpo_loss_type = "sigmoid"
    trainer.config.dpo_label_smoothing = 0.0
    for attr, val in (extra_config_attrs or {}).items():
        setattr(trainer.config, attr, val)
    with patch.object(Trainer, "__init__", _noop_trainer_init):
        cls.__init__(trainer)
    return trainer


class TestDPOTrainerDefaults:
    def test_beta_default(self):
        trainer = _make_dpo(DPOTrainer)
        assert trainer.beta == 0.1

    def test_loss_type_default(self):
        trainer = _make_dpo(DPOTrainer)
        assert trainer.loss_type == "sigmoid"

    def test_label_smoothing_default(self):
        trainer = _make_dpo(DPOTrainer)
        assert trainer.label_smoothing == 0.0

    def test_reference_log_probs_initially_none(self):
        trainer = _make_dpo(DPOTrainer)
        assert trainer._reference_chosen_log_probs is None
        assert trainer._reference_rejected_log_probs is None


class TestKTOTrainerInit:
    def test_loss_type_is_kto(self):
        trainer = _make_dpo(KTOTrainer, {"kto_beta": 0.1})
        assert trainer.loss_type == "kto"

    def test_beta_default(self):
        trainer = _make_dpo(KTOTrainer, {"kto_beta": 0.1})
        assert trainer.beta == 0.1

    def test_beta_override_from_config(self):
        trainer = _make_dpo(KTOTrainer, {"kto_beta": 0.5})
        assert trainer.beta == 0.5


class TestORPOTrainerInit:
    def test_loss_type_is_orpo(self):
        trainer = _make_dpo(ORPOTrainer, {"orpo_beta": 0.1})
        assert trainer.loss_type == "orpo"

    def test_beta_default(self):
        trainer = _make_dpo(ORPOTrainer, {"orpo_beta": 0.1})
        assert trainer.beta == 0.1

    def test_beta_override_from_config(self):
        trainer = _make_dpo(ORPOTrainer, {"orpo_beta": 0.25})
        assert trainer.beta == 0.25


class TestGRPOTrainerInit:
    def test_loss_type_is_grpo(self):
        trainer = _make_dpo(GRPOTrainer, {"grpo_beta": 0.04, "grpo_epsilon": 0.2, "grpo_num_generations": 4})
        assert trainer.loss_type == "grpo"

    def test_beta_default(self):
        trainer = _make_dpo(GRPOTrainer, {"grpo_beta": 0.04, "grpo_epsilon": 0.2, "grpo_num_generations": 4})
        assert trainer.beta == 0.04

    def test_epsilon_default(self):
        trainer = _make_dpo(GRPOTrainer, {"grpo_beta": 0.04, "grpo_epsilon": 0.2, "grpo_num_generations": 4})
        assert trainer.epsilon == 0.2

    def test_num_generations_default(self):
        trainer = _make_dpo(GRPOTrainer, {"grpo_beta": 0.04, "grpo_epsilon": 0.2, "grpo_num_generations": 4})
        assert trainer.num_generations == 4

    def test_override_all_grpo_params(self):
        trainer = _make_dpo(GRPOTrainer, {"grpo_beta": 0.1, "grpo_epsilon": 0.4, "grpo_num_generations": 8})
        assert trainer.beta == 0.1
        assert trainer.epsilon == 0.4
        assert trainer.num_generations == 8


class TestTrainerRegistration:
    def test_dpo_registered(self):
        assert "dpo" in get_llm_trainers_registry()

    def test_kto_registered(self):
        assert "kto" in get_llm_trainers_registry()

    def test_orpo_registered(self):
        assert "orpo" in get_llm_trainers_registry()

    def test_grpo_registered(self):
        assert "grpo" in get_llm_trainers_registry()

    def test_dpo_maps_to_dpo_trainer(self):
        assert get_llm_trainers_registry()["dpo"] is DPOTrainer

    def test_kto_maps_to_kto_trainer(self):
        assert get_llm_trainers_registry()["kto"] is KTOTrainer

    def test_orpo_maps_to_orpo_trainer(self):
        assert get_llm_trainers_registry()["orpo"] is ORPOTrainer

    def test_grpo_maps_to_grpo_trainer(self):
        assert get_llm_trainers_registry()["grpo"] is GRPOTrainer
