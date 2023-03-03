#!/bin/bash

# Download the data
wget https://ludwig.ai/latest/data/rotten_tomatoes.csv
wget https://ludwig.ai/latest/data/rotten_tomatoes_test.csv

# Check the first 5 rows
head -n 5 rotten_tomatoes.csv

# Train
python -m ludwig.cli train --config rotten_tomatoes.yaml --dataset rotten_tomatoes.csv

# Predict and Evaluate
python -m ludwig.cli predict --model_path results/experiment_run/model --dataset rotten_tomatoes_test.csv
