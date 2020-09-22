# -*- coding: utf-8 -*-
# Copyright (c) 2020 Uber Technologies, Inc.
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
import os
import shutil
import subprocess
import tempfile

import yaml

from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


def _run_ludwig(command, **ludwig_kwargs):
    commands = ['ludwig', command]
    for arg_name, value in ludwig_kwargs.items():
        commands += ['--' + arg_name, value]
    cmdline = ' '.join(commands)
    print(cmdline)
    exit_code = subprocess.call(cmdline, shell=True,
                                env=os.environ.copy())
    assert exit_code == 0


def _prepare_data(csv_filename, model_definition_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    dataset_filename = generate_data(input_features, output_features,
                                     csv_filename)

    # generate model definition file
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    with open(model_definition_filename, 'w') as f:
        yaml.dump(model_definition, f)

    return dataset_filename


def test_train_cli_dataset(csv_filename):
    """Test training using `ludwig train --dataset`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_definition_filename = os.path.join(tmpdir,
                                                 'model_definition.yaml')
        dataset_filename = _prepare_data(csv_filename,
                                         model_definition_filename)
        _run_ludwig('train',
                    dataset=dataset_filename,
                    model_definition_file=model_definition_filename,
                    output_directory=tmpdir)


def test_train_cli_training_set(csv_filename):
    """Test training using `ludwig train --training_set`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_definition_filename = os.path.join(tmpdir,
                                                 'model_definition.yaml')
        dataset_filename = _prepare_data(csv_filename,
                                         model_definition_filename)
        validation_filename = shutil.copyfile(
            dataset_filename, os.path.join(tmpdir, 'validation.csv'))
        test_filename = shutil.copyfile(
            dataset_filename, os.path.join(tmpdir, 'test.csv'))
        _run_ludwig('train',
                    training_set=dataset_filename,
                    validation_set=validation_filename,
                    test_set=test_filename,
                    model_definition_file=model_definition_filename,
                    output_directory=tmpdir)
