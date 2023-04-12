#!/usr/bin/env python

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic/).

# Import required libraries
import logging
import shutil

import pandas as pd
import yaml

from ludwig.api import LudwigModel

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

qa_pairs = {
    "Question": [
        "What is the capital of Uzbekistan?",
        "Who is the founder of Microsoft?",
        "What is the tallest building in the world?",
        "What is the currency of Brazil?",
        "What is the boiling point of mercury in Celsius?",
        "What is the most commonly spoken language in the world?",
        "What is the diameter of the Earth?",
        'Who wrote the novel "1984"?',
        "What is the name of the largest moon of Neptune?",
        "What is the speed of light in meters per second?",
        "What is the smallest country in Africa by land area?",
        "What is the largest organ in the human body?",
        'Who directed the film "The Godfather"?',
        "What is the name of the smallest planet in our solar system?",
        "What is the largest lake in Africa?",
        "What is the smallest country in Asia by land area?",
        "Who is the current president of Russia?",
        "What is the chemical symbol for gold?",
        "What is the name of the famous Swiss mountain known for skiing?",
        "What is the largest flower in the world?",
    ],
    "Answer": [
        "Tashkent",
        "Bill Gates",
        "Burj Khalifa",
        "Real",
        "-38.83",
        "Mandarin",
        "12,742 km",
        "George Orwell",
        "Triton",
        "299,792,458 m/s",
        "Seychelles",
        "Skin",
        "Francis Ford Coppola",
        "Mercury",
        "Lake Victoria",
        "Maldives",
        "Vladimir Putin",
        "Au",
        "The Matterhorn",
        "Rafflesia arnoldii",
    ],
}

df = pd.DataFrame(qa_pairs)

config = yaml.safe_load(
    """
        input_features:
            - name: Question
              type: text
        output_features:
            - name: Answer
              type: text
        model_type: llm
        model_name: hf-internal-testing/tiny-random-GPTJForCausalLM
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
    dataset=df, experiment_name="simple_experiment", model_name="simple_model", skip_save_processed_input=True
)

training_set, val_set, test_set, _ = preprocessed_data

# batch prediction
preds, _ = model.predict(test_set, skip_save_predictions=False)
print(preds)
