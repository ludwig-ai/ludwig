import os
from typing import List

import pytest
import yaml

from ludwig.config_validation.synthetic_validation import get_abrupt_config, validate_config_with_synthetic_data
from ludwig.utils.data_utils import load_yaml

BENCHMARK_DIRECTORY = "ludwig/benchmarking/configs"


def get_test_config_filenames() -> List[str]:
    """Return list of the config filenames used for benchmarking."""
    return [config_fp for config_fp in os.listdir(BENCHMARK_DIRECTORY)]


def test_get_abrupt_config():
    config = yaml.safe_load(
        """
output_features:
  - name: label
    type: category
input_features:
  - name: text
    type: text
    encoder:
      type: bert
preprocessing:
  text:
    max_sequence_length: 128
trainer:
  batch_size: 16
  learning_rate: 0.00002
  checkpoints_per_epoch: 2
  evaluate_training_set: False
  learning_rate_scheduler:
    warmup_evaluations: 0
  optimizer:
    type: adamw
  validation_field: label
  validation_metric: accuracy
  epochs: 1
    """
    )

    abrupt_config = get_abrupt_config(config)

    assert "bert" not in str(abrupt_config)


@pytest.mark.parametrize("config_filename", get_test_config_filenames())
def test_validate_config_with_synthetic_data(config_filename, tmpdir):
    config_path = os.path.join(BENCHMARK_DIRECTORY, config_filename)
    config = load_yaml(config_path)
    validate_config_with_synthetic_data(config)


def test_validate_config_with_synthetic_data_tagger_decoder():
    config = yaml.safe_load(
        """
input_features:
    -
        name: utterance
        type: text
        encoder:
            type: rnn
            cell_type: lstm
            reduce_output: null
        preprocessing:
          tokenizer: space

output_features:
    -
        name: tag
        type: sequence
        decoder:
            type: tagger
  """
    )

    validate_config_with_synthetic_data(config)


def test_validate_config_number_iq_normalization():
    config = yaml.safe_load(
        """
input_features:
  - name: TYPE
    type: binary
    column: TYPE
  - name: MARK_IV
    type: number
    preprocessing:
      normalization: iq
    column: MARK_IV
  - name: DELTA
    type: number
    preprocessing:
      normalization: null
    column: DELTA
  - name: VEGA
    type: number
    preprocessing:
      normalization: iq
    column: VEGA
  - name: normalized_width
    type: number
    preprocessing:
      normalization: iq
    column: normalized_width
  - name: bbo_quantity
    type: number
    preprocessing:
      normalization: iq
    column: bbo_quantity
  - name: TTE
    type: number
    preprocessing:
      normalization: zscore
    column: TTE
  - name: ABS_DELTA
    type: number
    preprocessing:
      normalization: zscore
    column: ABS_DELTA
output_features:
  - name: TRADE_COUNT
    type: number
    loss:
      type: huber
    column: TRADE_COUNT
trainer:
  validation_metric: mean_absolute_error
backend:
  type: ray
  processor:
    type: dask
    parallelism: 100
    persist: true
  """
    )

    validate_config_with_synthetic_data(config)
