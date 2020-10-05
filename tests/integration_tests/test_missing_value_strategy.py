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
import tempfile

import pandas as pd

from ludwig.api import LudwigModel
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data


def test_missing_value_prediction(csv_filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [category_feature(vocab_size=2, reduce_input='sum',
                                           preprocessing=dict(
                                               missing_value_strategy='fill_with_mode'))]
        output_features = [binary_feature()]

        dataset = pd.read_csv(
            generate_data(input_features, output_features, csv_filename))

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
        }
        model = LudwigModel(config)
        _, _, output_dir = model.train(dataset=dataset,
                                       output_directory=tmpdir)

        # Set the input column to None, we should be able to replace the missing value with the mode
        # from the training set
        dataset[input_features[0]['name']] = None
        model.predict(dataset=dataset)

        model = LudwigModel.load(os.path.join(output_dir, 'model'))
        model.predict(dataset=dataset)
