import json
import os
from typing import Optional, Type, Union

import pytest
import torch

import ludwig.schema.encoders.utils as schema_encoders_utils
from ludwig.api import LudwigModel
from ludwig.constants import ENCODER, MODEL_ECD, NAME, TEXT, TRAINER
from ludwig.encoders import text_encoders
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.schema.model_config import ModelConfig
from ludwig.utils.data_utils import load_json
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated
from tests.integration_tests.utils import category_feature, generate_data, HF_ENCODERS, LocalTestBackend, text_feature


def _load_pretrained_hf_model_no_weights(
    modelClass: Type,
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    **pretrained_kwargs,
):
    """Loads a HF model architecture without loading the weights."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    return AutoModel.from_config(config)


@pytest.fixture
def mock_load_encoder_from_hf_hub(monkeypatch):
    """Mocks encoder downloads from HuggingFace Hub.

    With this mock, only encoder configs are downloaded, not the encoder weights.
    """
    monkeypatch.setattr(text_encoders, "load_pretrained_hf_model", _load_pretrained_hf_model_no_weights)


def get_mismatched_config_params(ludwig_results_dir, ludwig_model):
    saved_config_dict = load_json(os.path.join(ludwig_results_dir, "model", MODEL_HYPERPARAMETERS_FILE_NAME))
    saved_config_obj = ModelConfig.from_dict(saved_config_dict)

    mismatches = []
    for input_feature_config in saved_config_obj.input_features.to_list():
        feature_name = input_feature_config[NAME]
        encoder_config_from_file = input_feature_config[ENCODER]
        encoder_config_from_model = ludwig_model.model.input_features[feature_name].encoder_obj.config.to_dict()
        for k, v in encoder_config_from_model.items():
            # Skip saved_weights_in_checkpoint because this value is not yet set when the global config
            # is modified with the final encoder config.
            if k == "saved_weights_in_checkpoint":
                continue

            if encoder_config_from_file[k] != v:
                mismatch = {
                    "feature_name": feature_name,
                    "param_name": k,
                    "val_from_file": encoder_config_from_file[k],
                    "val_from_model": v,
                }
                mismatches.append(mismatch)
    return mismatches


@pytest.mark.slow
@pytest.mark.parametrize("encoder_name", HF_ENCODERS)
def test_hf_ludwig_model_e2e(tmpdir, csv_filename, mock_load_encoder_from_hf_hub, encoder_name):
    """Tests HuggingFace encoders end-to-end.

    This test validates the following:
        1. Encoder config defaults are compatible with Ludwig experiments.
        2. Ludwig correctly updates the encoder config with the parameters introduced by the HF encoder.
        3. Ludwig correctly loads checkpoints containing HF encoder weights.
    """
    input_features = [
        text_feature(
            encoder={
                "vocab_size": 30,
                "min_len": 1,
                "type": encoder_name,
                "use_pretrained": True,
            }
        )
    ]
    output_features = [category_feature(decoder={"vocab_size": 2})]
    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"train_steps": 1},
    }
    model = LudwigModel(config=config, backend=LocalTestBackend())

    # Validates that the defaults associated with the encoder are compatible with Ludwig training.
    _, _, _, results_dir = model.experiment(dataset=rel_path, output_directory=tmpdir)

    # Validate that the saved config reflects the parameters introduced by the HF encoder.
    # This ensures that the config updates after initializing the encoder.
    mismatched_config_params = get_mismatched_config_params(results_dir, model)
    if len(mismatched_config_params) > 0:
        raise AssertionError(
            f"Config parameters mismatched with encoder parameters: {json.dumps(mismatched_config_params, indent=4)}"
        )

    # Validate the model can be loaded.
    # This ensures that the config reflects the internal architecture of the encoder.
    LudwigModel.load(os.path.join(results_dir, "model"))


@pytest.mark.parametrize("trainable", [True, False])
def test_distilbert_param_updates(trainable: bool):
    max_sequence_length = 20
    distil_bert_encoder = text_encoders.DistilBERTEncoder(
        use_pretrained=False,
        max_sequence_length=max_sequence_length,
        trainable=trainable,
    )

    # send a random input through the model with its initial weights
    inputs = torch.rand((2, max_sequence_length)).type(distil_bert_encoder.input_dtype)
    outputs = distil_bert_encoder(inputs)

    # perform a backward pass to update the model params
    target = torch.randn(outputs["encoder_output"].shape)
    check_module_parameters_updated(distil_bert_encoder, (inputs,), target)

    # send the same input through the model again. should be different if trainable, else the same
    outputs2 = distil_bert_encoder(inputs)

    encoder_output1 = outputs["encoder_output"]
    encoder_output2 = outputs2["encoder_output"]

    if trainable:
        # Outputs should be different if the model was updated
        assert not torch.equal(encoder_output1, encoder_output2)
    else:
        # Outputs should be the same if the model wasn't updated
        assert torch.equal(encoder_output1, encoder_output2)


@pytest.mark.parametrize("encoder_name", HF_ENCODERS)
def test_encoder_names_constant_synced_with_schema(encoder_name):
    """Ensures that each value in the HF_ENCODERS constant is represented by an equivalent schema object."""
    schema_encoders_utils.get_encoder_cls(MODEL_ECD, TEXT, encoder_name)
