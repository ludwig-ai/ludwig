#!/bin/bash

#
# Run 5-fold cross validation
#

ludwig experiment \
  --model_definition_file model_definition.yaml \
  --data_csv data/train.csv \
  --output_directory results \
  -kf 5

