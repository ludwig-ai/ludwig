import logging
import os

import yaml

from ludwig.api import LudwigModel

config = yaml.safe_load(
    """
input_features:
  - name: review
    type: text

    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: bert-base-uncased
      trainable: true
      adapter: lora

output_features:
  - name: sentiment
    type: category

preprocessing:
  type: random
  probabilities: [0.99, 0.005, 0.005]

trainer:
  batch_size: 8
  epochs: 2
  gradient_accumulation_steps: 8

backend:
  type: ray
  cache_dir: /src/cache
  trainer:
    use_gpu: true
    strategy:
      type: deepspeed
      zero_optimization:
        stage: 3
        offload_optimizer:
          device: cpu
          pin_memory: true
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
    dataset="ludwig://imdb",
    experiment_name="imdb_sentiment",
    model_name="bloom3b",
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
