import pytest

from ludwig.constants import BASE_MODEL, INPUT_FEATURES, MODEL_LLM, MODEL_TYPE, OUTPUT_FEATURES
from ludwig.schema.model_types.base import ModelConfig
from ludwig.utils.backend_utils import (
    _get_backend_type_from_config,
    _get_deepspeed_optimization_stage_from_config,
    _get_optimization_stage_from_trainer_config,
)
from tests.integration_tests.utils import text_feature


@pytest.fixture()
def model_config():
    return {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "yujiepan/llama-2-tiny-random",
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
        "generation": {"max_new_tokens": 8},
    }


@pytest.mark.parametrize("backend, expected_backend", [(None, "local"), ("local", "local"), ("ray", "ray")])
def test_get_backend_type_from_config(model_config, backend, expected_backend):
    if backend:
        model_config["backend"] = {"type": backend}
    model_config = ModelConfig.from_dict(model_config)
    assert _get_backend_type_from_config(model_config) == expected_backend


@pytest.mark.parametrize(
    "backend_config, expected_stage",
    [
        (None, None),
        ({"type": "local"}, None),
        ({"type": "ray"}, None),
        ({"type": "ray", "trainer": {"strategy": {"type": "ddp"}}}, None),
        ({"type": "ray", "trainer": {"strategy": {"type": "deepspeed"}}}, 3),
        ({"type": "ray", "trainer": {"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 0}}}}, 0),
        ({"type": "ray", "trainer": {"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 1}}}}, 1),
        ({"type": "ray", "trainer": {"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 2}}}}, 2),
        ({"type": "ray", "trainer": {"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 3}}}}, 3),
    ],
)
def test_get_deepspeed_optimization_stage_from_config(model_config, backend_config, expected_stage):
    if backend_config:
        model_config["backend"] = backend_config
    model_config = ModelConfig.from_dict(model_config)
    assert _get_deepspeed_optimization_stage_from_config(model_config) == expected_stage


@pytest.mark.parametrize(
    "backend_trainer_config, expected_stage",
    [
        ({"strategy": {"type": "ddp"}}, None),
        ({"strategy": {"type": "deepspeed"}}, 3),
        ({"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 0}}}, 0),
        ({"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 1}}}, 1),
        ({"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 2}}}, 2),
        ({"strategy": {"type": "deepspeed", "zero_optimization": {"stage": 3}}}, 3),
    ],
)
def test_get_optimization_stage_from_trainer_config(backend_trainer_config, expected_stage):
    assert _get_optimization_stage_from_trainer_config(backend_trainer_config) == expected_stage
