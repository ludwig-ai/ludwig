from ludwig.utils.state_dict_backward_compatibility import update_state_dict


def test_update_transformer_module_keys():
    state_dict_with_old_keys = {
        "input_features.module_dict.sentence__ludwig.encoder_obj.transformer.embeddings.LayerNorm.bias": 0.0,
        "sentence__ludwig.encoder_obj.transformer.encoder.layer.0.attention.output.LayerNorm.weight": 0.0,
        "module_dict.sentence__ludwig.encoder_obj.transformer.embeddings.word_embeddings.weight": 0.0,
    }

    expected_state_dict = {
        "input_features.module_dict.sentence__ludwig.encoder_obj.transformer.module.embeddings.LayerNorm.bias": 0.0,
        "sentence__ludwig.encoder_obj.transformer.module.encoder.layer.0.attention.output.LayerNorm.weight": 0.0,
        "module_dict.sentence__ludwig.encoder_obj.transformer.module.embeddings.word_embeddings.weight": 0.0,
    }

    # Ensures that, for models saved before FreezeModule was added, 'module' is added to the key path.
    updated_state_dict = update_state_dict(state_dict_with_old_keys)
    assert updated_state_dict == expected_state_dict


def test_does_not_update_freeze_module():
    state_dict = {
        "module_dict.sentence__ludwig.encoder_obj.transformer.module.embeddings.LayerNorm.bias": 0.0,
        "sentence__ludwig.encoder_obj.transformer.module.encoder.layer.0.attention.output.LayerNorm.weight": 0.0,
        "module_dict.sentence__ludwig.encoder_obj.transformer.module.embeddings.word_embeddings.weight": 0.0,
    }

    # Ensures that models saved with FreezeModule aren't modified.
    updated_state_dict = update_state_dict(state_dict)
    assert updated_state_dict == state_dict
