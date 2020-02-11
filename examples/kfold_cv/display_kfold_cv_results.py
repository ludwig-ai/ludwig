#!/usr/bin/env python
# coding: utf-8


import argparse
import os.path
import pprint
import sys

from ludwig.utils.data_utils import load_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Display K-fold cross validation results',
        prog='display_kfold_cv_results',
        usage='%(prog)s [options]'
    )

    # ----------------------------
    # Experiment naming parameters
    # ----------------------------
    parser.add_argument(
        '--results_directory',
        type=str,
        default='results',
        help='directory that contains the K-fold cv results'
    )

    args = parser.parse_args(sys.argv[1:])
    results_directory = args.results_directory

    print("Retrieving results from ", results_directory)

    kfold_cv_stats = load_json(
        os.path.join(results_directory, 'kfold_training_statistics.json')
    )

    print('#\n# K-fold Cross Validation Results\n#')
    pprint.pprint(kfold_cv_stats['overall'])
