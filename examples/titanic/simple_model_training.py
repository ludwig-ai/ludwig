#!/usr/bin/env python

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic/).

# Import required libraries
import logging
import os
import shutil

import numpy as np
import torch
from torch.autograd import Variable

from ludwig.api import LudwigModel
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.datasets import titanic
from ludwig.explain.captum import explain_ig
from ludwig.utils.neuropod_utils import generate_neuropod_torchscript

# clean out prior results
# shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
training_set, test_set, _ = titanic.load(split=True)
base_set = training_set  # .sample(n=100)
test_set = test_set.sample(n=10)
print(test_set)

# Define Ludwig model object that drive model training
# model = LudwigModel(config="./model1_config.yaml", logging_level=logging.INFO)

# # initiate model training
# (
#     train_stats,  # dictionary containing training statistics
#     preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
#     output_directory,  # location of training results stored on disk
# ) = model.train(dataset=training_set, experiment_name="simple_experiment", model_name="simple_model")

# # list contents of output directory
# print("contents of output directory:", output_directory)
# for item in os.listdir(output_directory):
#     print("\t", item)

model = LudwigModel.load("results/simple_experiment_simple_model/model")

# model_ts = generate_neuropod_torchscript(model)
# model_ts.save("titanic.pt")

# model_ts = torch.jit.load("titanic.pt")

# baseline = []
# for feature in model.model.input_features.values():
#     baseline.append(feature.create_sample_input())


# data_to_predict = [
#     torch.tensor(dataset.dataset[feature.proc_column].astype(np.float32), requires_grad=True)
#     for _, feature in model.model.input_features.items()
# ]
# # print(model_ts(*data_to_predict))
# print(data_to_predict)


attribution, expected_values, preds = explain_ig(model, inputs_df=test_set, sample_df=base_set, target="Survived")
print(attribution.shape)
print(np.array(expected_values).shape)
print(np.array(preds).shape)

# for (name, values), attribution in zip(inputs.items(), results):
#     print(name)
#     print(values)
#     print(attribution)
#     print(abs(attribution).mean(0))
#     print()

# for (name, values), encoded, attribution in zip(inputs.items(), encoded_inputs.values(), results):
#     print(name, values, encoded, attribution, abs(attribution).sum(0))

# restored_model = torch.jit.load("titanic.pt")

# batch prediction
# print(model.predict(test_set, skip_save_predictions=False))
