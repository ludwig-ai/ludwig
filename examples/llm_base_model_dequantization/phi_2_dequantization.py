import logging
import os

import yaml
from huggingface_hub import whoami

from ludwig.api import LudwigModel
from ludwig.utils.hf_utils import upload_folder_to_hfhub

hf_username = whoami().get("name")
base_model_name = "microsoft/phi-2"
dequantized_path = "microsoft-phi-2-dequantized"
save_path = "/home/ray/" + dequantized_path
hfhub_repo_id = os.path.join(hf_username, dequantized_path)


config = yaml.safe_load(
    f"""
    model_type: llm
    base_model: {base_model_name}

    quantization:
      bits: 4

    input_features:
      - name: instruction
        type: text

    output_features:
      - name: output
        type: text

    trainer:
        type: none

    backend:
      type: local
  """
)

# Define Ludwig model object that drive model training
model = LudwigModel(config=config, logging_level=logging.INFO)
model.save_dequantized_base_model(save_path=save_path)

# Optional: Upload to Huggingface Hub
upload_folder_to_hfhub(repo_id=hfhub_repo_id, folder_path=save_path)
