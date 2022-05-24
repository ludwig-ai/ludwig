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


def get_input_tensors(input_set):
    dataset, _ = preprocess_for_prediction(
        model.config,
        dataset=input_set,
        training_set_metadata=model.training_set_metadata,
        data_format="auto",
        split="full",
        include_outputs=False,
        backend=model.backend,
        callbacks=model.callbacks,
    )
    # print(dataset.get_dataset())

    inputs = {name: dataset.dataset[feature.proc_column] for name, feature in model.model.input_features.items()}
    return [torch.from_numpy(t) for t in inputs.values()], inputs, {}

    # encoded_inputs = model.model.encode(inputs)
    # # print("ENCODED_INPUTS", encoded_inputs)

    # # print(encoded_inputs)

    # data_to_predict = [v["encoder_output"] for _, v in encoded_inputs.items()]
    # # preds = model.model.predict_from_encoded(*[{"encoder_output": arg} for arg in data_to_predict])
    # # print(preds)

    # data_to_predict = [Variable(t, requires_grad=True) for t in data_to_predict]
    # return data_to_predict, inputs, encoded_inputs


def getmean(t):
    import random

    try:
        return t.mean()
    except:
        try:
            return torch.mode(t).values
        except:
            batch_size = t.shape[0]
            idx = random.randint(0, batch_size - 1)
            return t[idx]


data_to_predict, inputs, encoded_inputs = get_input_tensors(test_set)
baseline, _, _ = get_input_tensors(base_set)
baseline = [torch.unsqueeze(getmean(t), 0) for t in baseline]
# baseline[0] = torch.tensor([3], dtype=torch.int8)
print(baseline)

# baseline = []
# for feature in model.model.input_features.values():
#     baseline.append(feature.create_sample_input())


class WrapperModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model.model

    def forward(self, *args):
        # args = [{"encoder_output": arg} for arg in args]
        return model.model.predict_from_encoded(*args)[0]


model_fn = WrapperModule()
print("BASELINE PREDS", model_fn(*baseline))


# data_to_predict = [
#     torch.tensor(dataset.dataset[feature.proc_column].astype(np.float32), requires_grad=True)
#     for _, feature in model.model.input_features.items()
# ]
# # print(model_ts(*data_to_predict))
# print(data_to_predict)

from captum.attr import (
    configure_interpretable_embedding_layer,
    DeepLift,
    FeatureAblation,
    GradientShap,
    IntegratedGradients,
    LayerIntegratedGradients,
    NoiseTunnel,
    remove_interpretable_embedding_layer,
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

# print(data_to_predict)
# explainer = IntegratedGradients(model_fn)
# explainer = NoiseTunnel(explainer)
# explainer = DeepLift(model_fn)
# explainer = GradientShap(model_fn)
# explainer = FeatureAblation(model_fn)

layers = [layer for layer in model_fn.model.input_features.values()]
print(layers)
explainer = LayerIntegratedGradients(model_fn, layers, multiply_by_inputs=True)

results = explainer.attribute(
    tuple(data_to_predict),
    # n_steps=200,
    baselines=tuple(baseline),
    # method="gausslegendre",
    # return_convergence_delta=True,
)

results = [t.detach().numpy().sum(1) for t in results]

for (name, values), attribution in zip(inputs.items(), results):
    print(name)
    print(values)
    print(attribution)
    print(abs(attribution).mean(0))
    print()

# for (name, values), encoded, attribution in zip(inputs.items(), encoded_inputs.values(), results):
#     print(name, values, encoded, attribution, abs(attribution).sum(0))

# restored_model = torch.jit.load("titanic.pt")

# batch prediction
# print(model.predict(test_set, skip_save_predictions=False))
