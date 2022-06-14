from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

input_type_registry = Registry()


def register_input_feature(name: str):
    def wrap(cls):
        input_type_registry[name] = cls
        return cls
    return wrap


def get_input_feature_jsonschema():

    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": input_feature_types},
                "column": {"type": "string"},
                "encoder": {"type": "string"},
            },
            "allOf": get_input_encoder_conds(input_feature_types)
                     + get_input_preproc_conds(input_feature_types),
            "required": ["name", "type"],
        }
    }


def get_output_feature_jsonschema():

    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": output_feature_types},
                "column": {"type": "string"},
                "decoder": {"type": "string"},
            },
            "allOf": get_output_decoder_conds(output_feature_types)
            + get_output_preproc_conds(output_feature_types),
            "required": ["name", "type"],
        },
    }


def get_input_encoder_conds(input_feature_types):
    conds = []
    for feature_type in input_feature_types:
        encoder_names = list(get_encoder_classes(feature_type).keys())
        encoder_cond = create_cond(
            {"type": feature_type},
            {"encoder": {"enum": encoder_names}},
        )
        conds.append(encoder_cond)
    return conds


def get_input_preproc_conds(input_feature_types):
    conds = []
    for feature_type in input_feature_types:
        feature_cls = input_type_registry[feature_type]
        preproc_spec = {
            "type": "object",
            "properties": feature_cls.preprocessing_schema(),
            "additionalProperties": False,
        }
        preproc_cond = create_cond(
            {"type": feature_type},
            {"preprocessing": preproc_spec},
        )
        conds.append(preproc_cond)
    return conds
