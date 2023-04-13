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

df = pd.DataFrame(
    {
        "review": [
            "I loved this movie!",
            "The food was okay, but the service was terrible.",
            "I can't believe how rude the staff was.",
            "This book was a real page-turner.",
            "The hotel room was dirty and smelled bad.",
            "I had a great experience at this restaurant.",
            "The concert was amazing!",
            "The traffic was terrible on my way to work this morning.",
            "The customer service was excellent.",
            "I was disappointed with the quality of the product.",
            "The scenery on the hike was breathtaking.",
            "I had a terrible experience at this hotel.",
            "The coffee at this cafe was delicious.",
            "The weather was perfect for a day at the beach.",
            "I would definitely recommend this product.",
            "The wait time at the doctor's office was ridiculous.",
            "The museum was a bit underwhelming.",
            "I had a fantastic time at the amusement park.",
            "The staff at this store was extremely helpful.",
            "The airline lost my luggage and was very unhelpful.",
            "This album is a must-listen for any music fan.",
            "The food at this restaurant was just okay.",
            "I was pleasantly surprised by how great this movie was.",
            "The car rental process was quick and easy.",
            "The service at this hotel was top-notch.",
        ],
        "label": [
            "positive",
            "negative",
            "negative",
            "positive",
            "negative",
            "positive",
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "positive",
            "positive",
            "negative",
            "neutral",
            "positive",
            "positive",
            "negative",
            "positive",
            "neutral",
            "positive",
            "positive",
            "positive",
        ],
    }
)

config = yaml.safe_load(
    """
        input_features:
            - name: review
              type: text
        output_features:
            - name: label
              type: category
              preprocessing:
                labels: [positive, neutral, negative]
                fallback_label: neutral
                prompt_template: |
                    Context information is below.
                    ###
                    {review}
                    ###
                    Given the context information and not prior knowledge, classify the context as one of: {labels}
              decoder:
                type: parser
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
