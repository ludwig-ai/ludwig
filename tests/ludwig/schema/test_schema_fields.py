from ludwig.constants import (
    ENCODER,
    ENUM,
    INPUT_FEATURES,
    ITEMS,
    PROPERTIES,
    THEN,
    ALLOF,
)


def parse_schema(input_dict: dict, key_list: list):
    """Parse a nested dictionary by a list of keys."""
    parsed_dict = input_dict
    for key in key_list:
        parsed_dict = parsed_dict[key]
    return parsed_dict


def test_transformer_encoder_representations(ecd_schema):
    """Test that transformer encoder representation enum contains expected options."""

    input_feature_path = [PROPERTIES, INPUT_FEATURES, ITEMS, ALLOF]
    encoder_key_path = [THEN, PROPERTIES, ENCODER, ALLOF]
    representation_key_path = [THEN, PROPERTIES, "representation", ENUM]

    text_feature_schema = parse_schema(ecd_schema, input_feature_path)[10]
    text_transformer_encoder_schema = parse_schema(text_feature_schema, encoder_key_path)[7]
    text_transformer_encoder_enum = parse_schema(text_transformer_encoder_schema, representation_key_path)

    sequence_feature_schema = parse_schema(ecd_schema, input_feature_path)[8]
    sequence_transformer_encoder_schema = parse_schema(sequence_feature_schema, encoder_key_path)[7]
    sequence_transformer_encoder_enum = parse_schema(sequence_transformer_encoder_schema, representation_key_path)

    assert text_transformer_encoder_enum == sequence_transformer_encoder_enum == ["dense", "sparse"]


