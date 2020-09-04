# -*- coding: utf-8 -*-
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import multiprocessing
import os
import random
import shutil
import unittest
import uuid
from distutils.util import strtobool

import cloudpickle
import pandas as pd
from ludwig.constants import VECTOR
from ludwig.data.dataset_synthesizer import DATETIME_FORMATS
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.experiment import full_experiment

ENCODERS = [
    'embed', 'rnn', 'parallel_cnn', 'cnnrnn', 'stacked_parallel_cnn',
    'stacked_cnn'
]

HF_ENCODERS_SHORT = ['distilbert']

HF_ENCODERS = [
    'bert',
    'gpt',
    'gpt2',
    # 'transformer_xl',
    'xlnet',
    'xlm',
    'roberta',
    'distilbert',
    'ctrl',
    'camembert',
    'albert',
    't5',
    'xlmroberta',
    'longformer',
    'flaubert',
    'electra',
]


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError("If set, {} must be yes or no.".format(key))
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable
    to a truth value to run them.

    """
    if not _run_slow_tests:
        test_case = unittest.skip("Skipping: this test is too slow")(test_case)
    return test_case


def generate_data(
        input_features,
        output_features,
        filename='test_csv.csv',
        num_examples=25
):
    """
    Helper method to generate synthetic data based on input, output feature
    specs
    :param num_examples: number of examples to generate
    :param input_features: schema
    :param output_features: schema
    :param filename: path to the file where data is stored
    :return:
    """
    features = input_features + output_features
    df = build_synthetic_dataset(num_examples, features)
    data = [next(df) for _ in range(num_examples)]

    dataframe = pd.DataFrame(data[1:], columns=data[0])
    dataframe.to_csv(filename, index=False)

    return filename


def random_string(length=5):
    return uuid.uuid4().hex[:length].upper()


def numerical_feature(normalization=None):
    return {
        'name': 'num_' + random_string(),
        'type': 'numerical',
        'preprocessing': {
            'normalization': normalization
        }
    }


def category_feature(**kwargs):
    cat_feature = {
        'type': 'category',
        'name': 'category_' + random_string(),
        'vocab_size': 10,
        'embedding_size': 5
    }

    cat_feature.update(kwargs)
    return cat_feature


def text_feature(**kwargs):
    feature = {
        'name': 'text_' + random_string(),
        'type': 'text',
        'reduce_input': None,
        'vocab_size': 5,
        'min_len': 7,
        'max_len': 7,
        'embedding_size': 8,
        'state_size': 8
    }
    feature.update(kwargs)
    return feature


def set_feature(**kwargs):
    feature = {
        'type': 'set',
        'name': 'set_' + random_string(),
        'vocab_size': 10,
        'max_len': 5,
        'embedding_size': 5
    }
    feature.update(kwargs)
    return feature


def sequence_feature(**kwargs):
    seq_feature = {
        'type': 'sequence',
        'name': 'sequence_' + random_string(),
        'vocab_size': 10,
        'max_len': 7,
        'encoder': 'embed',
        'embedding_size': 8,
        'fc_size': 8,
        'state_size': 8,
        'num_filters': 8
    }
    seq_feature.update(kwargs)
    return seq_feature


def image_feature(folder, **kwargs):
    img_feature = {
        'type': 'image',
        'name': 'image_' + random_string(),
        'encoder': 'resnet',
        'preprocessing': {
            'in_memory': True,
            'height': 12,
            'width': 12,
            'num_channels': 3
        },
        'resnet_size': 8,
        'destination_folder': folder,
        'fc_size': 8,
        'num_filters': 8
    }
    img_feature.update(kwargs)
    return img_feature


def audio_feature(folder, **kwargs):
    feature = {
        'name': 'audio_' + random_string(),
        'type': 'audio',
        'preprocessing': {
            'audio_feature': {
                'type': 'fbank',
                'window_length_in_s': 0.04,
                'window_shift_in_s': 0.02,
                'num_filter_bands': 80
            },
            'audio_file_length_limit_in_s': 3.0
        },
        'encoder': 'stacked_cnn',
        'should_embed': False,
        'conv_layers': [
            {
                'filter_size': 400,
                'pool_size': 16,
                'num_filters': 32,
                'regularize': 'false'
            },
            {
                'filter_size': 40,
                'pool_size': 10,
                'num_filters': 64,
                'regularize': 'false'
            }
        ],
        'fc_size': 256,
        'audio_dest_folder': folder
    }
    feature.update(kwargs)
    return feature


def timeseries_feature(**kwargs):
    ts_feature = {
        'name': 'timeseries_' + random_string(),
        'type': 'timeseries',
        'max_len': 7
    }
    ts_feature.update(kwargs)
    return ts_feature


def binary_feature():
    return {
        'name': 'binary_' + random_string(),
        'type': 'binary'
    }


def bag_feature(**kwargs):
    feature = {
        'name': 'bag_' + random_string(),
        'type': 'bag',
        'max_len': 5,
        'vocab_size': 10,
        'embedding_size': 5
    }
    feature.update(kwargs)

    return feature


def date_feature(**kwargs):
    feature = {
        'name': 'date_' + random_string(),
        'type': 'date',
        'preprocessing': {
            'datetime_format': random.choice(list(DATETIME_FORMATS.keys()))
        }
    }

    feature.update(kwargs)

    return feature


def h3_feature(**kwargs):
    feature = {
        'name': 'h3_' + random_string(),
        'type': 'h3'
    }
    feature.update(kwargs)

    return feature


def vector_feature(**kwargs):
    feature = {
        'type': VECTOR,
        'vector_size': 5,
        'name': 'vector_' + random_string()
    }
    feature.update(kwargs)

    return feature


def run_experiment(input_features, output_features, **kwargs):
    """
    Helper method to avoid code repetition in running an experiment. Deletes
    the data saved to disk after running the experiment
    :param input_features: list of input feature dictionaries
    :param output_features: list of output feature dictionaries
    **kwargs you may also pass extra parameters to the experiment as keyword
    arguments
    :return: None
    """
    model_definition = None
    if input_features is not None and output_features is not None:
        # This if is necessary so that the caller can call with
        # model_definition_file (and not model_definition)
        model_definition = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {
                'type': 'concat',
                'fc_size': 14
            },
            'training': {'epochs': 2}
        }

    args = {
        'model_definition': model_definition,
        'skip_save_processed_input': True,
        'skip_save_progress': True,
        'skip_save_unprocessed_output': True,
        'skip_save_model': True,
        'skip_save_log': True
    }
    args.update(kwargs)

    exp_dir_name = full_experiment(**args)
    shutil.rmtree(exp_dir_name, ignore_errors=True)


def generate_output_features_with_dependencies(main_feature, dependencies):
    # helper function to generate multiple output features specifications
    # with dependencies, support for 'test_experiment_multiple_seq_seq` unit test
    # Parameters:
    # main_feature: feature identifier, valid values 'feat1', 'feat2', 'feat3'
    # dependencies: list of dependencies for 'main_feature', do not li
    # Example:
    #  generate_output_features_with_dependencies('feat2', ['feat1', 'feat3'])

    output_features = [
        category_feature(vocab_size=2, reduce_input='sum'),
        sequence_feature(vocab_size=10, max_len=5),
        numerical_feature()
    ]

    # value portion of dictionary is a tuple: (position, feature_name)
    #   position: location of output feature in the above output_features list
    #   feature_name: Ludwig generated feature name
    feature_names = {
        'feat1': (0, output_features[0]['name']),
        'feat2': (1, output_features[1]['name']),
        'feat3': (2, output_features[2]['name'])
    }

    # generate list of dependencies with real feature names
    generated_dependencies = [feature_names[feat_name][1]
                              for feat_name in dependencies]

    # specify dependencies for the main_feature
    output_features[feature_names[main_feature][0]]['dependencies'] = \
        generated_dependencies

    return output_features


def _subproc_wrapper(fn, queue, *args, **kwargs):
    fn = cloudpickle.loads(fn)
    results = fn(*args, **kwargs)
    queue.put(results)


def spawn(fn):
    def wrapped_fn(*args, **kwargs):
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()

        p = ctx.Process(
            target=_subproc_wrapper,
            args=(cloudpickle.dumps(fn), queue, *args),
            kwargs=kwargs)

        p.start()
        p.join()
        results = queue.get()
        return results

    return wrapped_fn
