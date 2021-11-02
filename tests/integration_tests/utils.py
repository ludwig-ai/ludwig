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
import logging
import multiprocessing
import os
import random
import shutil
import sys
import traceback
from typing import List
import unittest
import uuid
from distutils.util import strtobool
from typing import Union, Tuple

import cloudpickle
import numpy as np
import pandas as pd
import torch

import torch

from ludwig.api import LudwigModel
from ludwig.backend import LocalBackend
from ludwig.constants import VECTOR, COLUMN, NAME, PROC_COLUMN
from ludwig.data.dataset_synthesizer import DATETIME_FORMATS
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.experiment import experiment_cli
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.utils.data_utils import read_csv, replace_file_extension
from ludwig.utils.torch_utils import LudwigModule

logger = logging.getLogger(__name__)

# Used in sequence-related unit tests (encoders, features) as well as end-to-end integration tests.
# TODO(justin): Check for missing encoders.
ENCODERS = [
    'embed',
    'rnn',
    'parallel_cnn',
    'cnnrnn',
    'stacked_parallel_cnn',
    'stacked_cnn',
    'transformer'
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
    'mt5'
]


class LocalTestBackend(LocalBackend):
    @property
    def supports_multiprocessing(self):
        return False


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
        num_examples=25,

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


def numerical_feature(normalization=None, **kwargs):
    feature = {
        'name': 'num_' + random_string(),
        'type': 'numerical',
        'preprocessing': {
            'normalization': normalization
        }
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def category_feature(**kwargs):
    feature = {
        'type': 'category',
        'name': 'category_' + random_string(),
        'vocab_size': 10,
        'embedding_size': 5
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


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
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
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
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def sequence_feature(**kwargs):
    feature = {
        'type': 'sequence',
        'name': 'sequence_' + random_string(),
        'vocab_size': 10,
        'max_len': 7,
        'encoder': 'embed',
        'embedding_size': 8,
        'fc_size': 8,
        'state_size': 8,
        'num_filters': 8,
        'hidden_size': 8
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def image_feature(folder, **kwargs):
    feature = {
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
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


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
        'destination_folder': folder
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def timeseries_feature(**kwargs):
    feature = {
        'name': 'timeseries_' + random_string(),
        'type': 'timeseries',
        'max_len': 7
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def binary_feature(**kwargs):
    feature = {
        'name': 'binary_' + random_string(),
        'type': 'binary'
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def bag_feature(**kwargs):
    feature = {
        'name': 'bag_' + random_string(),
        'type': 'bag',
        'max_len': 5,
        'vocab_size': 10,
        'embedding_size': 5
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
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
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def h3_feature(**kwargs):
    feature = {
        'name': 'h3_' + random_string(),
        'type': 'h3'
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def vector_feature(**kwargs):
    feature = {
        'type': VECTOR,
        'vector_size': 5,
        'name': 'vector_' + random_string()
    }
    feature.update(kwargs)
    feature[COLUMN] = feature[NAME]
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def run_experiment(
        input_features,
        output_features,
        skip_save_processed_input=True,
        config=None,
        backend=None,
        **kwargs
):
    """
    Helper method to avoid code repetition in running an experiment. Deletes
    the data saved to disk after running the experiment
    :param input_features: list of input feature dictionaries
    :param output_features: list of output feature dictionaries
    **kwargs you may also pass extra parameters to the experiment as keyword
    arguments
    :return: None
    """
    if input_features is not None and output_features is not None:
        # This if is necessary so that the caller can call with
        # config_file (and not config)
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {
                'type': 'concat',
                'fc_size': 14
            },
            'training': {'epochs': 2}
        }

    args = {
        'config': config,
        'backend': backend or LocalTestBackend(),
        'skip_save_training_description': True,
        'skip_save_training_statistics': True,
        'skip_save_processed_input': skip_save_processed_input,
        'skip_save_progress': True,
        'skip_save_unprocessed_output': True,
        'skip_save_model': True,
        'skip_save_predictions': True,
        'skip_save_eval_stats': True,
        'skip_collect_predictions': True,
        'skip_collect_overall_stats': True,
        'skip_save_log': True
    }
    args.update(kwargs)

    _, _, _, _, exp_dir_name = experiment_cli(**args)
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
    try:
        results = fn(*args, **kwargs)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        results = e
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
        if isinstance(results, Exception):
            raise RuntimeError(
                f'Spawned subprocess raised {type(results).__name__}, '
                f'check log output above for stack trace.')
        return results

    return wrapped_fn


def get_weights(model: torch.nn.Module) -> List[torch.Tensor]:
    return [param.data for param in model.parameters()]


def run_api_experiment(input_features, output_features, data_csv):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(config)
    output_dir = None

    try:
        # Training with csv
        _, _, output_dir = model.train(
            dataset=data_csv,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        model.predict(dataset=data_csv)

        model_dir = os.path.join(output_dir, 'model')
        loaded_model = LudwigModel.load(model_dir)

        # Necessary before call to get_weights() to materialize the weights
        loaded_model.predict(dataset=data_csv)

        model_weights = get_weights(model.model)
        loaded_weights = get_weights(loaded_model.model)
        for model_weight, loaded_weight in zip(model_weights, loaded_weights):
            assert np.allclose(model_weight, loaded_weight)
    finally:
        # Remove results/intermediate data saved to disk
        shutil.rmtree(output_dir, ignore_errors=True)

    try:
        # Training with dataframe
        data_df = read_csv(data_csv)
        _, _, output_dir = model.train(
            dataset=data_df,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        model.predict(dataset=data_df)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def create_data_set_to_use(data_format, raw_data):
    # helper function for generating training and test data with specified format
    # handles all data formats except for hdf5
    # assumes raw_data is a csv dataset generated by
    # tests.integration_tests.utils.generate_data() function

    # support for writing to a fwf dataset based on this stackoverflow posting:
    # https://stackoverflow.com/questions/16490261/python-pandas-write-dataframe-to-fixed-width-file-to-fwf
    from tabulate import tabulate

    def to_fwf(df, fname):
        content = tabulate(df.values.tolist(), list(df.columns),
                           tablefmt="plain")
        open(fname, "w").write(content)

    pd.DataFrame.to_fwf = to_fwf

    dataset_to_use = None

    if data_format == 'csv':
        dataset_to_use = raw_data

    elif data_format in {'df', 'dict'}:
        dataset_to_use = pd.read_csv(raw_data)
        if data_format == 'dict':
            dataset_to_use = dataset_to_use.to_dict(orient='list')

    elif data_format == 'excel':
        dataset_to_use = replace_file_extension(raw_data, 'xlsx')
        pd.read_csv(raw_data).to_excel(
            dataset_to_use,
            index=False
        )

    elif data_format == 'excel_xls':
        dataset_to_use = replace_file_extension(raw_data, 'xls')
        pd.read_csv(raw_data).to_excel(
            dataset_to_use,
            index=False
        )

    elif data_format == 'feather':
        dataset_to_use = replace_file_extension(raw_data, 'feather')
        pd.read_csv(raw_data).to_feather(
            dataset_to_use
        )

    elif data_format == 'fwf':
        dataset_to_use = replace_file_extension(raw_data, 'fwf')
        pd.read_csv(raw_data).to_fwf(
            dataset_to_use
        )

    elif data_format == 'html':
        dataset_to_use = replace_file_extension(raw_data, 'html')
        pd.read_csv(raw_data).to_html(
            dataset_to_use,
            index=False
        )

    elif data_format == 'json':
        dataset_to_use = replace_file_extension(raw_data, 'json')
        pd.read_csv(raw_data).to_json(
            dataset_to_use,
            orient='records'
        )

    elif data_format == 'jsonl':
        dataset_to_use = replace_file_extension(raw_data, 'jsonl')
        pd.read_csv(raw_data).to_json(
            dataset_to_use,
            orient='records',
            lines=True
        )

    elif data_format == 'parquet':
        dataset_to_use = replace_file_extension(raw_data, 'parquet')
        pd.read_csv(raw_data).to_parquet(
            dataset_to_use,
            index=False
        )

    elif data_format == 'pickle':
        dataset_to_use = replace_file_extension(raw_data, 'pickle')
        pd.read_csv(raw_data).to_pickle(
            dataset_to_use
        )

    elif data_format == 'stata':
        dataset_to_use = replace_file_extension(raw_data, 'stata')
        pd.read_csv(raw_data).to_stata(
            dataset_to_use
        )

    elif data_format == 'tsv':
        dataset_to_use = replace_file_extension(raw_data, 'tsv')
        pd.read_csv(raw_data).to_csv(
            dataset_to_use,
            sep='\t',
            index=False
        )

    else:
        ValueError(
            "'{}' is an unrecognized data format".format(data_format)
        )

    return dataset_to_use


def train_with_backend(
        backend,
        config,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        predict=True,
        evaluate=True,
):
    model = LudwigModel(config, backend=backend)
    output_dir = None

    try:
        _, _, output_dir = model.train(
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True,
            skip_save_log=True,
        )

        if dataset is None:
            dataset = training_set

        if predict:
            preds, _ = model.predict(dataset=dataset)
            assert preds is not None

        if evaluate:
            _, eval_preds, _ = model.evaluate(dataset=dataset)
            assert eval_preds is not None

        return model
    finally:
        # Remove results/intermediate data saved to disk
        shutil.rmtree(output_dir, ignore_errors=True)


class ParameterUpdateError(Exception):
    pass


def assert_model_parameters_updated(
        model: LudwigModule,
        model_output: torch.Tensor,

) -> None:
    """
    Confirms that model parameters can be updated.
    Args:
        model: (LudwigModel) model to be tested.
        model_output: (torch.Tensor) output tensor from model

    Returns: None

    """
    # setup
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target_tensor = torch.randn(model_output.shape, dtype=model_output.dtype)

    # capture model parameters before doing parameter update pass
    before = [(x[0], x[1].clone()) for x in model.named_parameters()]

    # do update of model parameters
    loss = loss_function(model_output, target_tensor)
    loss.backward()
    optimizer.step()  # comment out to generate no update exception for testing

    # capture model parameters after one pass
    after = [(x[0], x[1].clone()) for x in model.named_parameters()]

    # check for parameter updates
    for b, a in zip(before, after):
        # ensure parameters are updated
        if not (a[1] != b[1]).any():
            raise ParameterUpdateError(
                f'Model parameter for {a[0]} did not update.  Before and after '
                f'contain same contents.\nBefore model update: {b[1]}\n'
                f'After module update: {a[1]}'
            )


def assert_model_parameters_updated_loop(
        model: LudwigModule,
        model_input_args: Tuple,
        max_steps: int = 4
) -> None:
    """
    Confirms that model parameters can be updated.
    Args:
        model: (LudwigModel) model to be tested.
        model_input_args: (tuple) input for model
        max_steps: (int) maximum number of steps allowed to test for parameter
            updates.

    Returns: None

    """
    # setup
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=100)

    # generate initial model output tensor
    model_output = model(*model_input_args)

    # create target tensor
    if isinstance(model_output, torch.Tensor):
        target_tensor = torch.randn(model_output.shape,
                                    dtype=model_output.dtype)
    elif isinstance(model_output, tuple):
        target_tensor = torch.randn(model_output[0].shape,
                                    dtype=model_output[0].dtype)
    else:
        raise RuntimeError(
            'Unable to setup target tensor for model parameter update testing.'
        )

    # todo: is test for all parameters eventually getting updated?
    # # capture model parameters before doing parameter update pass
    # before = [(x[0], x[1].clone()) for x in model.named_parameters()]

    step = 1
    while True:
        model.train(True)

        # make pass through model
        model_output = model(*model_input_args)

        # todo: this is for all model parameters are updated during a cycle
        # capture model parameters before doing parameter update pass
        before = [(x[0], x[1].clone()) for x in model.named_parameters()]

        # do update of model parameters
        if isinstance(model_output, torch.Tensor):
            loss = loss_function(model_output, target_tensor)
        else:
            loss = loss_function(model_output[0], target_tensor)
        loss.backward()
        optimizer.step()  # comment out to generate no update exception for testing

        # capture model parameters after one pass
        after = [(x[0], x[1].clone()) for x in model.named_parameters()]

        # check for parameter updates
        parameter_updated = []
        for b, a in zip(before, after):
            parameter_updated.append((a[1] != b[1]).any())

        # check to see if parameters were updated in all layers
        if all(parameter_updated):
            print(f'\n>>>> parameter updated at step {step}')
            break
        elif step >= max_steps:
            parameters_not_updated = []
            for updated, b, a in zip(parameter_updated, before, after):
                if not updated:
                    parameters_not_updated.append(
                        f'\nParameter {a[0]} not updated:\n'
                        f'\tbefore model forward() pass (requires grad:{b[1].requires_grad}): {b[1]}\n'
                        f'\tafter model forward() pass (requires grad:{a[1].requires_grad}): {a[1]}\n'
                    )
            raise ParameterUpdateError(
                f'Not all model parameters updated after {step} tries.\n'
                f'{"".join(parameters_not_updated)}'
            )

        step += 1
