#!/usr/bin/env python

# # Simple Model Training Example
#
# This is a simple example of how to use the LLM model type to train
# a model on a simple question and answer dataset. It uses the
# facebook/opt-350m model as the base LLM model.

# Import required libraries
import logging
import shutil

import pandas as pd
import yaml

from ludwig.api import LudwigModel

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

qa_pairs = [
    {"Question": "What is the capital of Uzbekistan?", "Answer": "Tashkent"},
    {"Question": "Who is the founder of Microsoft?", "Answer": "Bill Gates"},
    {"Question": "What is the tallest building in the world?", "Answer": "Burj Khalifa"},
    {"Question": "What is the currency of Brazil?", "Answer": "Real"},
    {"Question": "What is the boiling point of mercury in Celsius?", "Answer": "-38.83"},
    {"Question": "What is the most commonly spoken language in the world?", "Answer": "Mandarin"},
    {"Question": "What is the diameter of the Earth?", "Answer": "12,742 km"},
    {"Question": 'Who wrote the novel "1984"?', "Answer": "George Orwell"},
    {"Question": "What is the name of the largest moon of Neptune?", "Answer": "Triton"},
    {"Question": "What is the speed of light in meters per second?", "Answer": "299,792,458 m/s"},
    {"Question": "What is the smallest country in Africa by land area?", "Answer": "Seychelles"},
    {"Question": "What is the largest organ in the human body?", "Answer": "Skin"},
    {"Question": 'Who directed the film "The Godfather"?', "Answer": "Francis Ford Coppola"},
    {"Question": "What is the name of the smallest planet in our solar system?", "Answer": "Mercury"},
    {"Question": "What is the largest lake in Africa?", "Answer": "Lake Victoria"},
    {"Question": "What is the smallest country in Asia by land area?", "Answer": "Maldives"},
    {"Question": "Who is the current president of Russia?", "Answer": "Vladimir Putin"},
    {"Question": "What is the chemical symbol for gold?", "Answer": "Au"},
    {"Question": "What is the name of the famous Swiss mountain known for skiing?", "Answer": "The Matterhorn"},
    {"Question": "What is the largest flower in the world?", "Answer": "Rafflesia arnoldii"},
]

df = pd.DataFrame(qa_pairs)

config = yaml.safe_load("""
        input_features:
            - name: Question
              type: text
        output_features:
            - name: Answer
              type: text
        model_type: llm
        generation:
            temperature: 0.1
            top_p: 0.75
            top_k: 40
            num_beams: 4
            max_new_tokens: 5
        base_model: facebook/opt-350m
    """)

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
