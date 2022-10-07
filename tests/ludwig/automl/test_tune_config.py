import pytest

try:
    from ludwig.automl.auto_tune_config import reduce_text_feature_max_length
except ImportError:
    pass


@pytest.mark.distributed
def test_reduce_text_model_mem_99ptile():
    config = {"input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}]}
    training_set_metadata = {"description": {"max_sequence_length_99ptile": 117.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"max_sequence_length": 117}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd


@pytest.mark.distributed
def test_reduce_text_model_mem_128():
    config = {"input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}]}
    training_set_metadata = {"description": {"max_sequence_length_99ptile": 512.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"max_sequence_length": 128}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd


@pytest.mark.distributed
def test_reduce_text_model_mem_override():
    config = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"max_sequence_length": 256}},
    }
    training_set_metadata = {"description": {"max_sequence_length_99ptile": 117.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"max_sequence_length": 117}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd


@pytest.mark.distributed
def test_reduce_text_model_mem_respect():
    config = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"max_sequence_length": 56}},
    }
    training_set_metadata = {"description": {"max_sequence_length_99ptile": 117.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"max_sequence_length": 56}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd
