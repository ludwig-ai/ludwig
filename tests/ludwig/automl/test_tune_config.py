from ludwig.automl.auto_tune_config import reduce_text_feature_max_length


def test_reduce_text_model_mem_99ptile():
    config = {"input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}]}
    training_set_metadata = {"description": {"word_99ptile_max_sequence_length": 117.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"word_sequence_length_limit": 117}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd


def test_reduce_text_model_mem_128():
    config = {"input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}]}
    training_set_metadata = {"description": {"word_99ptile_max_sequence_length": 512.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"word_sequence_length_limit": 128}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd


def test_reduce_text_model_mem_override():
    config = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"word_sequence_length_limit": 256}},
    }
    training_set_metadata = {"description": {"word_99ptile_max_sequence_length": 117.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"word_sequence_length_limit": 117}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd


def test_reduce_text_model_mem_respect():
    config = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"word_sequence_length_limit": 56}},
    }
    training_set_metadata = {"description": {"word_99ptile_max_sequence_length": 117.0}}
    config_upd = {
        "input_features": [{"name": "description", "column": "description", "type": "text", "encoder": "bert"}],
        "preprocessing": {"text": {"word_sequence_length_limit": 56}},
    }
    reduce_text_feature_max_length(config, training_set_metadata)
    assert config == config_upd
