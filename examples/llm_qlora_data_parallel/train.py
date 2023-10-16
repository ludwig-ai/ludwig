import logging
import os

import numpy as np
import pandas as pd
import yaml

from ludwig.api import LudwigModel

np.random.seed(123)


# Llama-2-7b-hf requires HUGGING_FACE_HUB_TOKEN to be set as an environment variable
# You can get a token at https://huggingface.co/settings/tokens
if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    raise ValueError(
        "Please set your Hugging Face Hub token as an environment variable using `export "
        "HUGGING_FACE_HUB_TOKEN=your_token`. You can get a token at https://huggingface.co/settings/tokens"
    )

df = pd.read_json("https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json")

# We're going to create a new column called `split` where:
# 90% will be assigned a value of 0 -> train set
# 5% will be assigned a value of 1 -> validation set
# 5% will be assigned a value of 2 -> test set
# Calculate the number of rows for each split value
total_rows = len(df)
split_0_count = int(total_rows * 0.9)
split_1_count = int(total_rows * 0.05)
split_2_count = total_rows - split_0_count - split_1_count

# Create an array with split values based on the counts
split_values = np.concatenate([np.zeros(split_0_count), np.ones(split_1_count), np.full(split_2_count, 2)])

# Shuffle the array to ensure randomness
np.random.shuffle(split_values)

# Add the 'split' column to the DataFrame
df["split"] = split_values
df["split"] = df["split"].astype(int)

# For now, just use 1000 rows as a demonstration
df = df.head(n=1000)

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
        type: fixed
        column: split
    global_max_sequence_length: 512

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

model = LudwigModel(config=fine_tuning_config, logging_level=logging.INFO)
results = model.train(dataset=df)

(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    dataset=df,
    experiment_name="alpaca_instruct",
    model_name="llama2_7b",
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)

prediction_df = pd.DataFrame(
    [
        {"instruction": "Create an array of length 5 which contains all even numbers between 1 and 10.", "input": ""},
        {
            "instruction": "Create an array of length 15 containing numbers divisible by 3 up to 45.",
            "input": "",
        },
        {"instruction": "Create a nested loop to print every combination of numbers between 0-9", "input": ""},
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

print("Predict")
preds, _ = model.predict(dataset=prediction_df)
preds = preds.compute()
for input_with_prediction in zip(prediction_df["instruction"], prediction_df["input"], preds["output_response"]):
    print(f"Instruction: {input_with_prediction[0]}")
    print(f"Input: {input_with_prediction[1]}")
    print(f"Generated Output: {input_with_prediction[2][0]}")
    print("\n\n")
