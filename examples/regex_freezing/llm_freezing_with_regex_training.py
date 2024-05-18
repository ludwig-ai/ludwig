import logging

import yaml

from ludwig.api import LudwigModel

"""
To inspect model layers in the terminal, type: "ludwig collect_summary -pm facebook/opt-350m"

For some models, a HuggingFace Token will be necessary.
Once you obtain one, use "export HUGGING_FACE_HUB_TOKEN="<api_token>"" in the terminal.
"""

config_str = r"""
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
    encoder:
      trainable: true
    preprocessing:
      max_sequence_length: 256

output_features:
  - name: output
    type: text
    preprocessing:
      max_sequence_length: 256

trainer:
  type: finetune
  layers_to_freeze_regex: (decoder\.layers\.22\.*|decoder\.layers\.23\.*)
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    warmup_fraction: 0.01

preprocessing:
  sample_ratio: 0.1
"""

config = yaml.safe_load(config_str)

model = LudwigModel(config=config, logging_level=logging.INFO)
results = model.train(dataset="ludwig://alpaca")
for name, p in model.named_parameters():
    print(f"{name}, {p.requires_grad}")
print(results)
