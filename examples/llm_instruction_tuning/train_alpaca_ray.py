import logging
import os

import yaml

from ludwig.api import LudwigModel

config = yaml.safe_load(
    """
model_type: llm
model_name: facebook/opt-350m

input_features:
  - name: instruction
    type: text

output_features:
  - name: output
    type: text

# trainer:
#   batch_size: 8
#   epochs: 2
#   gradient_accumulation_steps: 8

trainer:
    type: finetune
    batch_size: auto

backend:
  type: local
#   type: ray
#   cache_dir: /src/cache
#   trainer:
#     use_gpu: true
#     strategy:
#       type: deepspeed
#       zero_optimization:
#         stage: 3
#         offload_optimizer:
#           device: cpu
#           pin_memory: true
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
    experiment_name="alpaca_instruct",
    model_name="bloom3b",
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
