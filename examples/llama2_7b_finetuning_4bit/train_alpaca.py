import logging
import os

import yaml

from ludwig.api import LudwigModel

config = yaml.safe_load("""
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

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
    learning_rate: 0.0003
    batch_size: 2
    gradient_accumulation_steps: 8
    epochs: 3
    learning_rate_scheduler:
      warmup_fraction: 0.01

backend:
  type: local
""")

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
    model_name="llama2_7b",
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
