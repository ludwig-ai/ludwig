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
from tests.integration_tests.utils import create_data_set_to_use, run_api_experiment
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


def test_dask_csv(csv_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=1000)
    dataset_parquet = create_data_set_to_use('parquet', dataset_csv)
    run_api_experiment(input_features, output_features, data_csv=dataset_parquet)
