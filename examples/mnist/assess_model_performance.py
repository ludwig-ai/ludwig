#!/usr/bin/env python
# coding: utf-8

# 
# Load a previously saved model and make predictions on the test data set
#

import os.path

# ## Import required libraries
import pandas as pd
from sklearn.metrics import accuracy_score

from ludwig.api import LudwigModel
from ludwig.datasets import mnist

# create data set for predictions
test_data = {'image_path': [], 'label': []}
dataset = mnist.Mnist()
test_dir = os.path.join(dataset.processed_dataset_path, 'testing')
for label in os.listdir(test_dir):
    files = os.listdir(os.path.join(test_dir, label))
    test_data['image_path'] += [os.path.join(test_dir, label, f) for f in
                                files]
    test_data['label'] += len(files) * [label]

# collect data into a data frame
test_df = pd.DataFrame(test_data)
print(test_df.head())

# retrieve a trained model
model = LudwigModel.load('./results/multiple_experiment_Option3/model')

# make predictions
pred_df, _ = model.predict(dataset=test_df)
print(pred_df.head())

# print accuracy on test data set
print('predicted accuracy', accuracy_score(test_df['label'], pred_df['label_predictions']))



