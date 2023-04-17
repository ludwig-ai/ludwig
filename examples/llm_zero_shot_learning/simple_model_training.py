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

logger = logging.getLogger(__name__)

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

review_label_pairs = [
    {"review": "I loved this movie!", "label": "positive", "split": 0},
    {"review": "The food was okay, but the service was terrible.", "label": "negative", "split": 0},
    {"review": "I can't believe how rude the staff was.", "label": "negative", "split": 0},
    {"review": "This book was a real page-turner.", "label": "positive", "split": 0},
    {"review": "The hotel room was dirty and smelled bad.", "label": "negative", "split": 0},
    {"review": "I had a great experience at this restaurant.", "label": "positive", "split": 0},
    {"review": "The concert was amazing!", "label": "positive", "split": 0},
    {"review": "The traffic was terrible on my way to work this morning.", "label": "negative", "split": 0},
    {"review": "The customer service was excellent.", "label": "positive", "split": 0},
    {"review": "I was disappointed with the quality of the product.", "label": "negative", "split": 0},
    {"review": "The scenery on the hike was breathtaking.", "label": "positive", "split": 0},
    {"review": "I had a terrible experience at this hotel.", "label": "negative", "split": 0},
    {"review": "The coffee at this cafe was delicious.", "label": "positive", "split": 0},
    {"review": "The weather was perfect for a day at the beach.", "label": "positive", "split": 0},
    {"review": "I would definitely recommend this product.", "label": "positive", "split": 1},
    {"review": "The wait time at the doctor's office was ridiculous.", "label": "negative", "split": 1},
    {"review": "The museum was a bit underwhelming.", "label": "neutral", "split": 1},
    {"review": "I had a fantastic time at the amusement park.", "label": "positive", "split": 1},
    {"review": "The staff at this store was extremely helpful.", "label": "positive", "split": 1},
    {"review": "This album is a must-listen for any music fan.", "label": "positive", "split": 1},
    {"review": "The food at this restaurant was just okay.", "label": "neutral", "split": 1},
    {"review": "I was pleasantly surprised by how great this movie was.", "label": "positive", "split": 2},
    {"review": "The car rental process was quick and easy.", "label": "positive", "split": 2},
    {"review": "The service at this hotel was top-notch.", "label": "positive", "split": 2},
    {"review": "The airline lost my luggage and was very unhelpful.", "label": "negative", "split": 2},
    {"review": "The food at the restaurant was okay", "label": "neutral", "split": 2},
]

df = pd.DataFrame(review_label_pairs)

config = yaml.safe_load(
    """
        input_features:
            - name: review
              type: text
        output_features:
            - name: label
              type: category
              preprocessing:
                vocab: [positive, neutral, negative]
                fallback_label: neutral
                prompt_template: |
                    Context information is below.
                    ###
                    {review}
                    ###
                    Without using prior knowledge, classify the review into one of these sentiment classes: {vocab}
              decoder:
                match:
                    positive:
                        type: contains
                        value: positive
                    neutral:
                        type: regex
                        value: neutral
                    negative:
                        type: contains
                        value: negative
        preprocessing:
            split:
                type: fixed
                column: split
        model_type: llm
        model_name: bigscience/bloomz-560m
        generation_config:
            temperature: 1.0
            num_beams: 4
            max_new_tokens: 10
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
    dataset=df,
    experiment_name="simple_experiment",
    model_name="simple_model",
    skip_save_processed_input=True,
)

# batch prediction
preds, _ = model.predict(df, skip_save_predictions=False, split="test")
logger.info(preds)
