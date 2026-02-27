#!/usr/bin/env python

# # Simple Model Training Example
#
# This is a simple example of how to use the LLM model type to train
# a zero shot classification model. It uses the facebook/opt-350m model
# as the base LLM model.

# Import required libraries
import logging
import shutil

import pandas as pd
import yaml

from ludwig.api import LudwigModel

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

review_label_pairs = [
    {"review": "I loved this movie!", "label": "positive"},
    {"review": "The food was okay, but the service was terrible.", "label": "negative"},
    {"review": "I can't believe how rude the staff was.", "label": "negative"},
    {"review": "This book was a real page-turner.", "label": "positive"},
    {"review": "The hotel room was dirty and smelled bad.", "label": "negative"},
    {"review": "I had a great experience at this restaurant.", "label": "positive"},
    {"review": "The concert was amazing!", "label": "positive"},
    {"review": "The traffic was terrible on my way to work this morning.", "label": "negative"},
    {"review": "The customer service was excellent.", "label": "positive"},
    {"review": "I was disappointed with the quality of the product.", "label": "negative"},
    {"review": "The scenery on the hike was breathtaking.", "label": "positive"},
    {"review": "I had a terrible experience at this hotel.", "label": "negative"},
    {"review": "The coffee at this cafe was delicious.", "label": "positive"},
    {"review": "The weather was perfect for a day at the beach.", "label": "positive"},
    {"review": "I would definitely recommend this product.", "label": "positive"},
    {"review": "The wait time at the doctor's office was ridiculous.", "label": "negative"},
    {"review": "The museum was a bit underwhelming.", "label": "neutral"},
    {"review": "I had a fantastic time at the amusement park.", "label": "positive"},
    {"review": "The staff at this store was extremely helpful.", "label": "positive"},
    {"review": "The airline lost my luggage and was very unhelpful.", "label": "negative"},
    {"review": "This album is a must-listen for any music fan.", "label": "positive"},
    {"review": "The food at this restaurant was just okay.", "label": "neutral"},
    {"review": "I was pleasantly surprised by how great this movie was.", "label": "positive"},
    {"review": "The car rental process was quick and easy.", "label": "positive"},
    {"review": "The service at this hotel was top-notch.", "label": "positive"},
]

df = pd.DataFrame(review_label_pairs)
df["split"] = [0] * 15 + [2] * 10

config = yaml.safe_load("""
model_type: llm
base_model: facebook/opt-350m
generation:
    temperature: 0.1
    top_p: 0.75
    top_k: 40
    num_beams: 4
    max_new_tokens: 64
prompt:
    task: "Classify the sample input as either negative, neutral, or positive."
    retrieval:
        type: semantic
        k: 3
        model_name: paraphrase-MiniLM-L3-v2
input_features:
-
    name: review
    type: text
output_features:
-
    name: label
    type: category
    preprocessing:
        fallback_label: "neutral"
    decoder:
        type: category_extractor
        match:
            "negative":
                type: contains
                value: "positive"
            "neural":
                type: contains
                value: "neutral"
            "positive":
                type: contains
                value: "positive"
preprocessing:
    split:
        type: fixed
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
