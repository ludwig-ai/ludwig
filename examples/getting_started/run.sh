#!/usr/bin/env bash

# Fail fast if an error occurs
set -e

# Get the directory of this script, which contains the config file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Download the data
wget https://ludwig.ai/latest/data/rotten_tomatoes.csv
wget https://ludwig.ai/latest/data/rotten_tomatoes_test.csv

# Check the first 5 rows
head -n 5 rotten_tomatoes.csv

# Train
ludwig train --config ${SCRIPT_DIR}/rotten_tomatoes.yaml --dataset rotten_tomatoes.csv

# Predict and Evaluate
ludwig predict --model_path results/experiment_run/model --dataset rotten_tomatoes_test.csv
