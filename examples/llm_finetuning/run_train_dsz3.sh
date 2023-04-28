#!/usr/bin/env bash

# Fail fast if an error occurs
set -e

# Get the directory of this script, which contains the config file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Train
deepspeed --num_gpus 4 ludwig train --config ${SCRIPT_DIR}/imdb_deepspeed_zero3.yaml --dataset ludwig://imdb
