import logging
import os

import yaml

from ludwig.api import LudwigModel

config = yaml.safe_load(
    """
model_type: llm
base_model: bigscience/bloomz-3b

quantization:
  bits: 4

adapter:
  type: lora

input_features:
  - name: instruction
    type: text

output_features:
  - name: output
    type: text

trainer:
    type: finetune
    batch_size: 4
    epochs: 3

backend:
  type: local
"""
)

# Define Ludwig model object that drive model training
model = LudwigModel(config=config, logging_level=logging.INFO)

# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    dataset="ludwig://alpaca",
    experiment_name="alpaca_instruct_4bit",
    model_name="bloom560m",
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
