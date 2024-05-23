import logging

import yaml

from ludwig.api import LudwigModel

"""
To inspect model layers in the terminal, type: "ludwig collect_summary -pm resnet18"

For some models, a HuggingFace Token will be necessary.
Once you obtain one, use "export HUGGING_FACE_HUB_TOKEN="<api_token>"" in the terminal.
"""

config_str = yaml.safe_load(
    r"""
model_type: llm
base_model: facebook/opt-350m

adapter:
  type: lora

prompt:
  template: |
    ### Instruction:
    Generate a concise summary of the following text, capturing the main points and conclusions.

    ### Input:
    {input}

    ### Response:

input_features:
  - name: prompt
    type: text
    preprocessing:
      max_sequence_length: 256


output_features:
  - name: output
    type: text
    preprocessing:
      max_sequence_length: 256

trainer:
  type: finetune
  layers_to_freeze_regex: (decoder\.layers\.22\.final_layer_norm\.*)
  learning_rate: 0.0001
  batch_size: 5
  gradient_accumulation_steps: 16
  epochs: 1
  learning_rate_scheduler:
    warmup_fraction: 0.01

preprocessing:
  sample_ratio: 0.1

generation:
  pad_token_id : 0
"""
)

model = LudwigModel(config=config_str, logging_level=logging.INFO)
results = model.train(dataset="ludwig://alpaca")
