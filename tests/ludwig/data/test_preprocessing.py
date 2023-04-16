import pandas as pd

from ludwig.schema.model_config import ModelConfig
from ludwig.data.preprocessing import is_input_feature, format_data_with_prompt
from tests.integration_tests.utils import text_feature, category_feature, generate_data_as_dataframe


def test_is_input_feature():
    # Adds encoder when output_feature=False
    assert is_input_feature(text_feature(output_feature=False)) is True
    # Adds decoder when output_feature=True
    assert is_input_feature(text_feature(output_feature=True)) is False


def test_format_data_with_prompt():
    prompt_config = {
        "task": (
            "Given the sample input, complete this sentence by replacing XXXX: "
            "The label is XXXX. Choose one value in this list: {label}."
        ),
        "retrieval": {
            "type": "index",
            "model_name": "multi-qa-MiniLM-L6-cos-v1",  # TODO(geoffrey): find a smaller model for testing
            "k": 2,
        },
    }

    input_features = [text_feature(name="description", encoder={"type": "passthrough"})]
    output_features = [category_feature(
        name="label", 
        output_feature=True, 
        decoder={"type": "category_parser"}, 
        preprocessing={"vocab": ["1","2","3"]},
    )]
    
    config = ModelConfig.from_dict({
        'model_type': 'llm',
        'model_name': 'eachadea/vicuna-13b-1.1',
        'input_features': input_features,
        'output_features': output_features,
    })

    df = generate_data_as_dataframe(input_features, output_features, 10)
    
    dataset_cols = df.to_dict(orient="list")
    feature_configs = input_features + output_features
    new_dataset_cols = format_data_with_prompt(dataset_cols, prompt_config, feature_configs)
    
    assert dataset_cols.keys() == new_dataset_cols.keys()
    assert len(dataset_cols["description"]) == len(new_dataset_cols["description"])
    
    # Inspect the generated prompts
    for prompt in new_dataset_cols["description"]:
        # TODO(geoffrey): this is brittle– we should come up with a better way for checking the prompt for values.
        # May consider starting each section off with some re-usable constant strings i.e. "CONTEXT:", "SAMPLE INPUT:", etc.
        assert "context" in prompt
        assert "sample input" in prompt
        assert "XXXX" in prompt
        assert "['1', '2', '3']" in prompt
