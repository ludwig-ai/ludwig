#!/usr/bin/env python
# coding: utf-8


# Download and prepare training data set
# Create Ludwig config file
#
# Based on the [UCI Wisconsin Breast Cancer data set](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))
#
import os.path
import shutil

import pandas as pd
import requests
import yaml
from sklearn.model_selection import train_test_split

# Constants
DATA_SET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
DATA_SET = 'wdbc.data'
DATA_DIR = './data'
RESULTS_DIR = 'results'

# Clean out previous results
print("Cleaning out old results")
if os.path.isfile(DATA_SET):
    os.remove(DATA_SET)
if os.path.isfile('config.yaml'):
    os.remove('config.yaml')

shutil.rmtree(RESULTS_DIR, ignore_errors=True)
shutil.rmtree(DATA_DIR, ignore_errors=True)

# Retrieve data from UCI Machine Learning Repository
# Download required data
print("Downloading data set")
r = requests.get(DATA_SET_URL)
if r.status_code == 200:
    with open(DATA_SET, 'w') as f:
        f.write(r.content.decode("utf-8"))

# create pandas dataframe from downloaded data
print("Preparing data for training")
raw_df = pd.read_csv(DATA_SET,
                     header=None,
                     sep=",", skipinitialspace=True)
raw_df.columns = ['ID', 'diagnosis'] + ['X' + str(i) for i in range(1, 31)]

# convert diagnosis attribute to binary format
raw_df['diagnosis'] = raw_df['diagnosis'].map({'M': 1, 'B': 0})

# Create train/test split
print("Saving training and test data sets")
train_df, test_df = train_test_split(raw_df, train_size=0.8, random_state=17)
os.mkdir(DATA_DIR)
train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)

print("Preparing Ludwig config")
# Create ludwig input_features
num_features = ['X' + str(i) for i in range(1, 31)]
input_features = []

# setup input features for numerical variables
for p in num_features:
    a_feature = {'name': p, 'type': 'numerical',
                 'preprocessing': {'missing_value_strategy': 'fill_with_mean',
                                   'normalization': 'zscore'}}
    input_features.append(a_feature)

# Create ludwig output features
output_features = [
    {
        'name': 'diagnosis',
        'type': 'binary',
        'num_fc_layers': 2,
        'fc_size': 64
    }
]

# setup ludwig config
config = {
    'input_features': input_features,
    'output_features': output_features,
    'training': {
        'epochs': 20,
        'batch_size': 32
    }
}

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

print("Completed data preparation")
