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
from ludwig.utils.neuropod_utils import generate_neuropod_torchscript

# clean out prior results
# shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
training_set, test_set, _ = titanic.load(split=True)

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

dataset, _ = preprocess_for_prediction(
    model.config,
    dataset=test_set,
    training_set_metadata=model.training_set_metadata,
    data_format="auto",
    split="full",
    include_outputs=False,
    backend=model.backend,
    callbacks=model.callbacks,
)
# print(dataset.get_dataset())

inputs = {name: dataset.dataset[feature.proc_column] for name, feature in model.model.input_features.items()}
encoded_inputs = model.model.encode(inputs)
# print(encoded_inputs)

print(encoded_inputs)

data_to_predict = [v["encoder_output"] for _, v in encoded_inputs.items()]
# preds = model.model.predict_from_encoded(*[{"encoder_output": arg} for arg in data_to_predict])
# print(preds)

data_to_predict = [Variable(t, requires_grad=True) for t in data_to_predict]


def model_fn(*args):
    args = [{"encoder_output": arg} for arg in args]
    return model.model.predict_from_encoded(*args)[0]


# data_to_predict = [
#     torch.tensor(dataset.dataset[feature.proc_column].astype(np.float32), requires_grad=True)
#     for _, feature in model.model.input_features.items()
# ]
# # print(model_ts(*data_to_predict))
# print(data_to_predict)

from captum.attr import IntegratedGradients

print(data_to_predict)
ig = IntegratedGradients(model_fn)
results = ig.attribute(
    tuple(data_to_predict),
    #  baselines=(baseline1, baseline2),
    method="gausslegendre",
    # return_convergence_delta=True,
)
print(results)

# restored_model = torch.jit.load("titanic.pt")

# batch prediction
# print(model.predict(test_set, skip_save_predictions=False))
