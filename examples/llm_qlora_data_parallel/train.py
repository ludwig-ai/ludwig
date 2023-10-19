import logging
import os

import numpy as np
import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import code_alpaca

np.random.seed(123)


# Llama-2-7b-hf requires HUGGING_FACE_HUB_TOKEN to be set as an environment variable
# You can get a token at https://huggingface.co/settings/tokens
if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    raise ValueError(
        "Please set your Hugging Face Hub token as an environment variable using `export "
        "HUGGING_FACE_HUB_TOKEN=your_token`. You can get a token at https://huggingface.co/settings/tokens"
    )

fine_tuning_config = yaml.safe_load(
    """
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

input_features:
    - name: instruction
      type: text

output_features:
    - name: output
      type: text

prompt:
    template: >-
        Below is an instruction that describes a task, paired with an input
        that provides further context. Write a response that appropriately
        completes the request.

        ### Instruction: {instruction}

        ### Input: {input}

        ### Response:

generation:
    temperature: 0.1
    max_new_tokens: 256

adapter:
    type: lora

quantization:
    bits: 4

preprocessing:
    split:
        type: random
        probabilities:
        - 0.9
        - 0.05
        - 0.05
    global_max_sequence_length: 512
    sample_size: 1000

backend:
    type: ray
    trainer:
        use_gpu: true
        strategy:
            type: deepspeed
            zero_optimization:
                stage: 2

trainer:
    type: finetune
    epochs: 3
    batch_size: 1
    eval_batch_size: 1
    enable_gradient_checkpointing: true
    gradient_accumulation_steps: 4
    learning_rate: 0.0001
    learning_rate_scheduler:
        decay: cosine
        warmup_fraction: 0.03
    """
)

df = code_alpaca.load(split=False)
model = LudwigModel(config=fine_tuning_config, logging_level=logging.INFO)

(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    dataset=df,
    experiment_name="code_alpaca_instruct",
    model_name="llama2_7b",
)

# List contents of output directory
print("Contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)

# Run Inference
print("Predict")
prediction_df = pd.DataFrame(
    [
        {
            "instruction": "Create an array of length 5 which contains all even numbers between 1 and 10.",
            "input": "",
        },
        {
            "instruction": "Create an array of length 15 containing numbers divisible by 3 up to 45.",
            "input": "",
        },
        {
            "instruction": "Create a nested loop to print every combination of numbers between 0-9",
            "input": "",
        },
        {
            "instruction": "Generate a function that computes the sum of the numbers in a given list",
            "input": "",
        },
        {
            "instruction": "Create a class to store student names, ages and grades.",
            "input": "",
        },
        {
            "instruction": "Print out the values in the following dictionary.",
            "input": "my_dict = {\n  'name': 'John Doe',\n  'age': 32,\n  'city': 'New York'\n}",
        },
    ]
)
preds, _ = model.predict(dataset=prediction_df)
preds = preds.compute()
for input_with_prediction in zip(prediction_df["instruction"], prediction_df["input"], preds["output_response"]):
    print(f"Instruction: {input_with_prediction[0]}")
    print(f"Input: {input_with_prediction[1]}")
    print(f"Generated Output: {input_with_prediction[2][0]}")
    print("\n\n")
