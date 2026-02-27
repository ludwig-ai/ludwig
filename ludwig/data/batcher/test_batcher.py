import logging

import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.data.dataset.pandas import PandasDataset


def test_pandas_size():
    df = pd.DataFrame(
        {"name": ["joe", "janice", "sara"], "mask": ["green", "black", "pink"], "weapon": ["stick", "gun", "gun"]}
    )
    config = yaml.safe_load("""
    model_type: llm
    base_model: HuggingFaceH4/tiny-random-LlamaForCausalLM
    input_features:
    - name: name
      type: text
      preprocessing:
        max_sequence_length: 256
      column: name
    output_features:
    - name: weapon
      type: text
      preprocessing:
        max_sequence_length: 256
      column: weapon
    preprocessing:
      split:
        type: random
        probabilities:
          - 1
          - 0
          - 0
    """)
    model = LudwigModel(config=config, logging_level=logging.INFO)
    data = model.preprocess(df, skip_save_processed_input=False)
    training_set = data[0]
    assert training_set.size == len(df)

    # Check if string loading works as well
    # data[0].data_hdf5_fp is the string filepath to the cached data from preprocessing
    data_from_str = PandasDataset(data[0].data_hdf5_fp, data[0].features, None)
    assert data_from_str.size == len(df)


def test_pandas_batcher_use_all_samples():
    df = pd.DataFrame(
        {"name": ["joe", "janice", "sara"], "mask": ["green", "black", "pink"], "weapon": ["stick", "gun", "gun"]}
    )
    config = yaml.safe_load("""
    model_type: llm
    base_model: HuggingFaceH4/tiny-random-LlamaForCausalLM
    input_features:
    - name: name
      type: text
      preprocessing:
        max_sequence_length: 256
      column: name
    output_features:
    - name: weapon
      type: text
      preprocessing:
        max_sequence_length: 256
      column: weapon
    preprocessing:
      split:
        type: random
        probabilities:
          - 1
          - 0
          - 0
    """)
    model = LudwigModel(config=config, logging_level=logging.INFO)
    data = model.preprocess(df, skip_save_processed_input=False)
    training_set = data[0]
    features = training_set.dataset.keys()

    batches = []
    with training_set.initialize_batcher(batch_size=1) as batcher:
        while not batcher.last_batch():
            batch = batcher.next_batch()
            batches.append(batch)
    assert (len(batches)) == training_set.size

    # Check to see if all items are used exactly once
    for feature in features:
        for i in range(len(training_set.dataset[feature])):
            # Each of the arrays in the line below should contain the vector representation of a feature of sample i
            assert (batches[i][feature].squeeze() == training_set.dataset[feature][i].squeeze()).all()

    # Check if string loading works as well
    batches = []
    # data[0].data_hdf5_fp is the string filepath to the cached data from preprocessing
    data_from_str = PandasDataset(data[0].data_hdf5_fp, data[0].features, None)
    features = data_from_str.dataset.keys()

    with data_from_str.initialize_batcher(batch_size=1) as batcher:
        while not batcher.last_batch():
            batch = batcher.next_batch()
            batches.append(batch)
    assert (len(batches)) == data_from_str.size

    # Check to see if all items are used exactly once
    for feature in features:
        for i in range(len(data_from_str.dataset[feature])):
            # Each of the arrays in the line below should contain the vector representation of a feature of sample i
            assert (batches[i][feature].squeeze() == data_from_str.dataset[feature][i].squeeze()).all()
