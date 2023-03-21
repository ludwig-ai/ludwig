from explore_schema import (
    combine_configs,
    combine_configs_for_comparator_combiner,
    combine_configs_for_sequence_combiner,
)

TABULAR_TO_TEXT = """
input_features:
  - name: category_1
    type: category
  - name: number_1
    type: number
  - name: binary_1
    type: binary
output_features:
  - name: text_output_1
    type: text
"""

TABULAR = """
input_features:
  - name: category_1
    type: category
  - name: number_1
    type: number
  - name: binary_1
    type: binary
output_features:
  - name: category_output_1
    type: category
"""

TEXT_INPUT = """
input_features:
  - name: text_1
    type: text
    preprocessing:
      max_sequence_length: 8
output_features:
  - name: category_output_1
    type: category
"""

NUMBER_INPUT = """
input_features:
  - name: number_1
    type: number
output_features:
  - name: category_output_1
    type: category
"""

CATEGORY_INPUT = """
input_features:
  - name: category_1
    type: category
output_features:
  - name: category_output_1
    type: category
"""

BINARY_INPUT = """
input_features:
  - name: binary_1
    type: binary
output_features:
  - name: category_output_1
    type: category
"""

TEXT_OUTPUT = """
input_features:
  - name: text_1
    type: text
    preprocessing:
      max_sequence_length: 8
output_features:
  - name: text_output_1
    type: text
"""

NUMBER_OUTPUT = """
input_features:
  - name: number_1
    type: number
output_features:
  - name: number_output_1
    type: number
"""

CATEGORY_OUTPUT = """
input_features:
  - name: category_1
    type: category
output_features:
  - name: category_output_1
    type: category
"""

BINARY_OUTPUT = """
input_features:
  - name: binary_1
    type: binary
output_features:
  - name: binary_output_1
    type: binary
"""

feature_type_to_config_for_encoder_preprocessing = {
    "number": NUMBER_INPUT,
    "category": CATEGORY_INPUT,
    "binary": BINARY_INPUT,
    "text": TEXT_INPUT,
}

feature_type_to_config_for_decoder_loss = {
    "number": NUMBER_OUTPUT,
    "category": CATEGORY_OUTPUT,
    "binary": BINARY_OUTPUT,
    "text": TEXT_OUTPUT,
}

ecd_config_section_to_config = {
    "trainer": TABULAR,
    "comparator": TABULAR,
    "concat": TABULAR,
    "project_aggregate": TABULAR,
    "sequence": TEXT_INPUT,
    "sequence_concat": TEXT_INPUT,
    "tabnet": TABULAR,
    "tabtransformer": TABULAR,
    "transformer": TABULAR,
}

combiner_type_to_combine_config_fn = {
    "comparator": combine_configs_for_comparator_combiner,
    "concat": combine_configs,
    "project_aggregate": combine_configs,
    "sequence": combine_configs_for_sequence_combiner,
    "sequence_concat": combine_configs_for_sequence_combiner,
    "tabnet": combine_configs,
    "tabtransformer": combine_configs,
    "transformer": combine_configs,
}
