#!/bin/bash

#
# Run 5-fold cross validation
#

ludwig experiment \
  --model_definition_file model_definition.yaml \
  --data_csv data/train.csv \
  --output_directory results \
  -kf 5

#
# Display results from k-fold cv
#
./display_kfold_cv_results.py --results_directory results