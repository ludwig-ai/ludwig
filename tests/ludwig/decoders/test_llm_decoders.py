import pytest
import torch

from ludwig.constants import (BACKEND, BASE_MODEL, GENERATION, INPUT_FEATURES,
                              MODEL_TYPE, OUTPUT_FEATURES)
from ludwig.decoders.llm_decoders import TextExtractorDecoder
from ludwig.schema.model_config import ModelConfig
from tests.integration_tests.utils import text_feature

TEST_MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"


def test_text_extractor_decoder():
    max_new_tokens = 4

    input_features = [
        {
            "name": "Question",
            "type": "text",
            "encoder": {"type": "passthrough"},
        }
    ]
    output_features = [text_feature(output_feature=True, name="Answer", decoder={"type": "text_extractor"})]

    config = {
        MODEL_TYPE: "llm",
        BASE_MODEL: TEST_MODEL_NAME,
        GENERATION: {
            "temperature": 0.1,
            "top_p": 0.75,
            "top_k": 40,
            "num_beams": 4,
            "max_new_tokens": max_new_tokens,
        },
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        BACKEND: "local",
    }

    config = ModelConfig.from_dict(config)
    decoder_config = config.output_features[0].decoder

    decoder = TextExtractorDecoder(32, decoder_config)

    inputs = [
        torch.tensor([1, 1, 1, 2, 2, 2, 2]),  # baseline
        torch.tensor([1, 1, 1, 2]),  # too short; test padding
        torch.tensor([1, 1, 1, 1, 2, 2, 2]),  # test different input length
    ]
    input_lengths = [3, 3, 4]

    # tests happy path
    outputs = decoder.forward(inputs, input_lengths, max_new_tokens)
    assert outputs["predictions"].shape == (3, max_new_tokens)
    # Create a Boolean mask for elements equal to 0 or 2 (padding or output)
    mask = (outputs["predictions"] == 0) | (outputs["predictions"] == 2)
    assert mask.all()

    # test overly long generation fails without updated max_new_tokens
    inputs.append(torch.tensor([1, 1, 1, 2, 2, 2, 2, 2]))  # too long; test downstream failure)
    input_lengths.append(3)
    with pytest.raises(ValueError):
        outputs = decoder.forward(inputs, input_lengths, max_new_tokens)

    # test overly long generation succeeds with new max_new_tokens
    new_max_new_tokens = 5
    outputs = decoder.forward(inputs, input_lengths, new_max_new_tokens)
    assert outputs["predictions"].shape == (4, new_max_new_tokens)
    mask = (outputs["predictions"] == 0) | (outputs["predictions"] == 2)
    assert mask.all()
