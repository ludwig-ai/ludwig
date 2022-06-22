# Copyright (c) 2022 Predibase, Inc.
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

from ludwig.api import LudwigModel
from ludwig.modules import serialization
from tests.integration_tests.utils import category_feature, generate_data, sequence_feature


def test_serialize_simple_model(tmpdir):
    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]

    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
    }

    model_1 = LudwigModel(config)
    model_1.train(dataset=data_csv, output_directory=tmpdir)

    saved_path = os.path.join(tmpdir, "model.h5")
    serialization.save(model_1.model, saved_path)
    state = serialization.load_state_from_file(saved_path)
    assert state is not None


if __name__ == "__main__":
    import pytest

    pytest.main(["-k", "test_serialize_simple_model"])
