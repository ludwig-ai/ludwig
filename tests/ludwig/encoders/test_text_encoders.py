import os

import pytest
import requests
import torch
from unittest.mock import patch

from ludwig.api import LudwigModel
from ludwig.constants import ENCODER, NAME, TRAINER
from ludwig.encoders import text_encoders
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.schema.encoders import text_encoders as configs
from ludwig.schema.model_config import ModelConfig
from ludwig.utils.data_utils import load_json
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated
from tests.integration_tests.utils import (
    category_feature,
    text_feature,
    generate_data,
    LocalTestBackend,
)

@pytest.mark.parametrize(
    "encoder_config_cls",
    [
        configs.ALBERTConfig,
        configs.BERTConfig,
        configs.XLMConfig,
        configs.GPTConfig,
        configs.RoBERTaConfig,
        configs.GPT2Config,
        configs.DistilBERTConfig,
        configs.TransformerXLConfig,
        configs.CTRLConfig,
        configs.CamemBERTConfig,
        configs.MT5Config,
        configs.XLMRoBERTaConfig,
        configs.LongformerConfig,
        configs.ELECTRAConfig,
        configs.FlauBERTConfig,
        configs.T5Config,
        configs.XLNetConfig,
        configs.DistilBERTConfig,
    ],
)
def test_hf_pretrained_default_exists(encoder_config_cls: configs.SequenceEncoderConfig):
    """Test that the default pretrained model exists on the HuggingFace Hub.

    This test merely checks that the default model name is valid. It does not check
    the model end-to-end, as that would require downloading the model weights, which
    can cause problems in the CI due to memory/runtime constraints.

    TODO: add an end-to-end test for pretrained HF encoders.
    """
    from huggingface_hub import HfApi

    default_model = encoder_config_cls.pretrained_model_name_or_path

    try:
        hf_api = HfApi()
        hf_api.model_info(default_model)
    except requests.exceptions.HTTPError:
        assert (
            False
        ), f"Unable to find model info for the default model '{default_model}' of config '{encoder_config_cls}'."


def load_pretrained_hf_model_no_weights(
    modelClass: Type, pretrained_model_name_or_path: Optional[Union[str, PathLike]], **pretrained_kwargs
) -> PreTrainedModel:
    """Download a HuggingFace model.
    Downloads a model from the HuggingFace zoo with retry on failure.
    Args:
        model: Class of the model to download.
    Returns:
        The pretrained model object.
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    return modelClass.from_config(config)


@pytest.mark.parametrize(
    "encoder_config_cls",
    [
        # configs.ALBERTConfig,
        # configs.BERTConfig,
        # configs.XLMConfig,
        # configs.GPTConfig,
        # configs.RoBERTaConfig,
        # configs.GPT2Config,
        # configs.DistilBERTConfig,
        # configs.TransformerXLConfig,
        # configs.CTRLConfig,
        configs.CamemBERTConfig,
        # configs.MT5Config,
        # configs.XLMRoBERTaConfig,
        # configs.LongformerConfig,
        # configs.ELECTRAConfig,
        # configs.FlauBERTConfig,
        # configs.T5Config,
        # configs.XLNetConfig,
        # configs.DistilBERTConfig,
    ],
)
def test_hf_ludwig_model_e2e(csv_filename, encoder_config_cls):
    tmpdir = "/Users/geoffreyangus/Downloads/hf_test_3"
    input_features = [text_feature(encoder={"vocab_size": 30, "min_len": 1, "type": encoder_config_cls.type, "use_pretrained": True})]
    output_features = [category_feature(decoder={"vocab_size": 2})]
    rel_path = generate_data(input_features, output_features, csv_filename)
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"train_steps": 1},
    }
    
    with patch("ludwig.utils.hf_utils.load_pretrained_hf_model", wraps=load_pretrained_hf_model_no_weights) as load_pretrained_hf_model_mock:
        model = LudwigModel(config=config, backend=LocalTestBackend())
        _, _, results_dir = model.train(dataset=rel_path, output_directory=tmpdir)
        assert load_pretrained_hf_model_mock.assert_called_once()
    
    # Validate that the model config reflects the parameters introduced by the HF encoder.
    # This ensures that the config updates after initializing the encoder.
    rendered_config_dict = load_json(os.path.join(results_dir, "model", MODEL_HYPERPARAMETERS_FILE_NAME))
    rendered_config_obj = ModelConfig.from_dict(rendered_config_dict)
    for input_feature_config in rendered_config_obj.input_features.to_list():
        input_feature_name = input_feature_config[NAME]
        encoder_obj_config = model.model.input_features[input_feature_name].encoder_obj.config.to_dict()
        for k, v in encoder_obj_config.items():
            assert input_feature_config[ENCODER][k] == v, (
                f"Encoder config mismatch for {k} in {input_feature_name}. Expected {v}, got {encoder_obj_config[k]}.")
    
    # Validate the model can be loaded.
    # This ensures that the config reflects the internal architecture of the encoder.
    LudwigModel.load(os.path.join(results_dir, "model"))
    
    
@pytest.fixture(scope="module")
def auto_transformer_tmpdir(tmpdir_factory):
    """Creates a temporary directory for `test_auto_transformer_encoder` to eliminate redundant downloads."""
    return tmpdir_factory.mktemp("auto_transformer")


@pytest.mark.parametrize("pretrained_model_name_or_path", [configs.AutoTransformerConfig.pretrained_model_name_or_path])
@pytest.mark.parametrize("reduce_output", [configs.AutoTransformerConfig.reduce_output, None])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_auto_transformer_encoder(
    auto_transformer_tmpdir, pretrained_model_name_or_path: str, reduce_output: str, max_sequence_length: int
):
    """Tests that loading an auto-transformer encoder works end-to-end.

    This has a separate test because AutoTransformerEncoder.DEFAULT_MODEL_NAME is None.
    """
    encoder = text_encoders.AutoTransformerEncoder(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
        pretrained_kwargs={"cache_dir": auto_transformer_tmpdir},
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.ALBERTConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_albert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    albert_encoder = text_encoders.ALBERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(albert_encoder.input_dtype)
    outputs = albert_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == albert_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.BERTConfig.reduce_output, None, "cls_pooled", "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_bert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    bert = text_encoders.BERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(bert.input_dtype)
    outputs = bert(inputs)
    assert outputs["encoder_output"].shape[1:] == bert.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.XLMConfig.reduce_output, "last", "sum", "mean"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_xlm_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    xlm_encoder = text_encoders.XLMEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(xlm_encoder.input_dtype)
    outputs = xlm_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == xlm_encoder.output_shape


@pytest.mark.skip(reason="Causes exit code 143 in CI")
@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.GPTConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_gpt_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    gpt_encoder = text_encoders.GPTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(gpt_encoder.input_dtype)
    outputs = gpt_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == gpt_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.RoBERTaConfig.reduce_output, "cls_pooled", "sum", None])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_roberta_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    roberta_encoder = text_encoders.RoBERTaEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(roberta_encoder.input_dtype)
    outputs = roberta_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == roberta_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.GPT2Config.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_gpt2_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    gpt_encoder = text_encoders.GPT2Encoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(gpt_encoder.input_dtype)
    outputs = gpt_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == gpt_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.DistilBERTConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_distil_bert(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    distil_bert_encoder = text_encoders.DistilBERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(distil_bert_encoder.input_dtype)
    outputs = distil_bert_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == distil_bert_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.TransformerXLConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_transfoxl_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    transfo = text_encoders.TransformerXLEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.randint(10, (2, max_sequence_length)).type(transfo.input_dtype)
    outputs = transfo(inputs)
    assert outputs["encoder_output"].shape[1:] == transfo.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.CTRLConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_ctrl_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.CTRLEncoder(
        max_sequence_length,
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.CamemBERTConfig.reduce_output, None, "cls_pooled"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_camembert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.CamemBERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.MT5Config.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_mt5_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    mt5_encoder = text_encoders.MT5Encoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(mt5_encoder.input_dtype)
    outputs = mt5_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == mt5_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.XLMRoBERTaConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_xlmroberta_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    xlmroberta_encoder = text_encoders.XLMRoBERTaEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(xlmroberta_encoder.input_dtype)
    outputs = xlmroberta_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == xlmroberta_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.LongformerConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_longformer_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.LongformerEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.ELECTRAConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_electra_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.ELECTRAEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.FlauBERTConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_flaubert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.FlauBERTEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.T5Config.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_t5_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.T5Encoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [configs.XLNetConfig.reduce_output, None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_xlnet_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    xlnet_encoder = text_encoders.XLNetEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(xlnet_encoder.input_dtype)
    outputs = xlnet_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == xlnet_encoder.output_shape


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
