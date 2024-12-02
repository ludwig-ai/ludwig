from ludwig.config_sampling.explore_schema import (
    combine_configs,
    combine_configs_for_comparator_combiner,
    combine_configs_for_sequence_combiner,
)

# A generic tabular to text config used to generate synthetic data and train a model on it.
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

# A generic tabular config used to generate synthetic data and train a model on it.
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

# A generic config with a single text input feature used to generate synthetic data and train a model on it.
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

# A generic config with a single number input feature used to generate synthetic data and train a model on it.
NUMBER_INPUT = """
input_features:
  - name: number_1
    type: number
output_features:
  - name: category_output_1
    type: category
"""

# A generic config with a single category input feature used to generate synthetic data and train a model on it.
CATEGORY_INPUT = """
input_features:
  - name: category_1
    type: category
output_features:
  - name: category_output_1
    type: category
"""

# A generic config with a single binary input feature used to generate synthetic data and train a model on it.
BINARY_INPUT = """
input_features:
  - name: binary_1
    type: binary
output_features:
  - name: category_output_1
    type: category
"""

# A generic config with a text output feature used to generate synthetic data and train a model on it.
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

# A generic config with a number output feature used to generate synthetic data and train a model on it.
NUMBER_OUTPUT = """
input_features:
  - name: number_1
    type: number
output_features:
  - name: number_output_1
    type: number
"""

# A generic config with a category output feature used to generate synthetic data and train a model on it.
CATEGORY_OUTPUT = """
input_features:
  - name: category_1
    type: category
output_features:
  - name: category_output_1
    type: category
"""

# A generic config with a binary output feature used to generate synthetic data and train a model on it.
BINARY_OUTPUT = """
input_features:
  - name: binary_1
    type: binary
output_features:
  - name: binary_output_1
    type: binary
"""

# Dictionary that maps from feature type to base config used to test the encoder and preprocessing sections.
FEATURE_TYPE_TO_CONFIG_FOR_ENCODER_PREPROCESSING = {
    "number": NUMBER_INPUT,
    "category": CATEGORY_INPUT,
    "binary": BINARY_INPUT,
    "text": TEXT_INPUT,
}

# Dictionary that maps from feature type to base config used to test the decoder and loss sections.
FEATURE_TYPE_TO_CONFIG_FOR_DECODER_LOSS = {
    "number": NUMBER_OUTPUT,
    "category": CATEGORY_OUTPUT,
    "binary": BINARY_OUTPUT,
    "text": TEXT_OUTPUT,
}

# Dictionary that maps from config section to base config used to test that section.
ECD_CONFIG_SECTION_TO_CONFIG = {
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

# Dictionary that maps from the combiner type to base config used to test that combiner.
COMBINER_TYPE_TO_COMBINE_FN_MAP = {
    "comparator": combine_configs_for_comparator_combiner,
    "concat": combine_configs,
    "project_aggregate": combine_configs,
    "sequence": combine_configs_for_sequence_combiner,
    "sequence_concat": combine_configs_for_sequence_combiner,
    "tabnet": combine_configs,
    "tabtransformer": combine_configs,
    "transformer": combine_configs,
}
