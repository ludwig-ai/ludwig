#!/bin/bash

#
# Download and prepare training data
#
python prepare_classification_data_set.py

#
# Run 5-fold cross validation
#
ludwig experiment \
  --model_definition_file model_definition.yaml \
  --data_csv data/train.csv \
  --output_directory results \
  --logging_level 'error' \
  -kf 5

#
# Display results from K-fold cv
#
python display_kfold_cv_results.py --results_directory results