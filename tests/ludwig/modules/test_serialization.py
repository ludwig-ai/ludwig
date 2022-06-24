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

import numpy as np

from ludwig.api import LudwigModel
from ludwig.modules import serialization
from ludwig.modules.ludwig_module import LudwigModuleState
from tests.integration_tests.utils import category_feature, generate_data, sequence_feature, text_feature


def assert_module_states_equal(a: LudwigModuleState, b: LudwigModuleState):
    """Recursively asserts that module states a and b are identical."""
    assert a.type == b.type
    assert a.ludwig_version == b.ludwig_version
    assert a.config == b.config
    assert sorted(a.saved_weights.keys()) == sorted(b.saved_weights.keys())
    for ak, av in a.saved_weights.items():
        assert ak in b.saved_weights
        bv = b.saved_weights[ak]
        assert np.allclose(av, bv)
    assert sorted(a.children.keys()) == sorted(b.children.keys())
    for ak, ac in a.children.items():
        assert ak in b.children
        bc = b.children[ak]
        assert_module_states_equal(ac, bc)


def test_serialize_deserialize_encoder(tmpdir):
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

    # Get pre-trained encoder
    trained_input_feature = model_1.model.input_features[input_features[0]["name"]]
    input_feature_encoder = trained_input_feature.encoder_obj

    encoder_state = input_feature_encoder.get_state()
    assert encoder_state is not None

    restored_encoder = serialization.instantiate_module_from_state(encoder_state, device="cpu")
    assert isinstance(restored_encoder, type(input_feature_encoder))

    restored_encoder_state = restored_encoder.get_state()
    assert restored_encoder_state is not None
    assert_module_states_equal(encoder_state, restored_encoder_state)


def test_load_save_encoder(tmpdir):
    input_features = [text_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]

    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
    }

    model_1 = LudwigModel(config)
    model_1.train(dataset=data_csv, output_directory=tmpdir)
    # Get pre-trained encoder
    trained_input_feature = model_1.model.input_features[input_features[0]["name"]]
    input_feature_encoder = trained_input_feature.encoder_obj

    saved_path = os.path.join(tmpdir, "text_encoder.h5")
    serialization.save(input_feature_encoder, saved_path)

    # Attempt to load model from saved file
    restored_encoder = serialization.load(saved_path, "cpu")
    assert restored_encoder is not None

    # TODO: construct new model with previously trained encoder


if __name__ == "__main__":
    import pytest

    pytest.main(["-k", "test_load_save_encoder"])
