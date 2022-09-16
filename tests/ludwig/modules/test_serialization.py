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
import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.modules import serialization
from ludwig.modules.ludwig_module import LudwigModuleState
from tests.integration_tests.utils import category_feature, generate_data, sequence_feature, text_feature


def assert_module_states_equal(a: LudwigModuleState, b: LudwigModuleState):
    """Recursively asserts that module states a and b are identical."""
    assert a.type == b.type
    assert a.ludwig_version == b.ludwig_version
    assert a.config == b.config
    assert a.metadata == b.metadata
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
    # Gets pre-trained encoder.
    trained_input_feature = model_1.model.input_features[input_features[0]["name"]]
    input_feature_encoder = trained_input_feature.encoder_obj
    # Get pre-trained encoder state, instantiates a new encoder instance from it.
    encoder_state = input_feature_encoder.get_state()
    restored_encoder = serialization.instantiate_module_from_state(encoder_state, device="cpu")
    restored_encoder_state = restored_encoder.get_state()
    # Ensures restored encoder's state is identical to pre-trained encoder's state.
    assert encoder_state is not None
    assert isinstance(restored_encoder, type(input_feature_encoder))
    assert restored_encoder_state is not None
    assert_module_states_equal(encoder_state, restored_encoder_state)


def test_load_save_encoder(tmpdir):
    torch.random.manual_seed(17)
    input_features = [text_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=5)]
    text_input_name = input_features[0]["name"]  # Auto-generated from random number by text_feature
    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))
    model1_config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 10},
    }
    model1 = LudwigModel(model1_config)
    train_stats1, _, _ = model1.train(dataset=data_csv, output_directory=tmpdir)
    # Saves pre-trained encoder to file.
    trained_input_feature = model1.model.input_features[text_input_name]
    input_feature_encoder = trained_input_feature.encoder_obj
    saved_path = os.path.join(tmpdir, "text_encoder.h5")
    serialization.save(input_feature_encoder, saved_path)
    # Ensures that we can restore encoder from saved path.
    restored_encoder = serialization.load(saved_path, "cpu")
    # Creates new model referencing the pre-trained encoder.
    model2_config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 20},
    }
    model2_config["input_features"][0]["encoder"]["pretrained_model"] = f"file://{saved_path}"
    model2 = LudwigModel(model2_config)
    train_stats2, _, _ = model2.train(dataset=data_csv, output_directory=tmpdir)
    assert restored_encoder is not None


def test_transfer_learning(tmpdir):
    torch.random.manual_seed(17)
    # Trains model 1 on generated dataset.
    input_features = [text_feature(reduce_output="sum", vocab_size=16, embedding_size=32)]
    output_features = [category_feature(vocab_size=5)]
    text_input_name = input_features[0]["name"]  # Auto-generated from random number by text_feature(...)
    category_output_name = output_features[0]["name"]  # Auto-generated from random number by category_feature(...)
    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"), 100)
    model1_config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
    }
    model1 = LudwigModel(model1_config)
    train_stats1, _, _ = model1.train(dataset=data_csv, output_directory=os.path.join(tmpdir, "model1"))
    # Saves pre-trained encoder to file.
    trained_input_feature = model1.model.input_features[text_input_name]
    input_feature_encoder = trained_input_feature.encoder_obj
    saved_path = os.path.join(tmpdir, "text_encoder.h5")
    # Note: I'd like to make the preprocessing metadata a property of the encoder, instead of having to get it from
    # training_set_metadata and pass it in as it is here.
    # I'd also like to call it preprocessing_data - I'd make that change in a separate PR which merges into this one.
    serialization.save(
        input_feature_encoder, saved_path, metadata=model1.training_set_metadata[trained_input_feature.feature_name]
    )
    # Trains model 2 on new dataset with different input column name.
    original_training_set = pd.read_csv(data_csv)
    new_training_set = original_training_set.rename(columns={text_input_name: "text_column"})
    new_training_set = pd.concat(
        [
            new_training_set,
            pd.DataFrame(
                {
                    "text_column": ["some new words", "which don't appear in the vocab", "should be mapped to UNK"],
                    category_output_name: ["test", "test", "test"],
                }
            ),
        ],
        axis=0,
    )
    model2_config = {
        "input_features": [{"type": "text", "name": "text_column"}],
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
    }
    model2_config["input_features"][0]["encoder"] = {"pretrained_model": f"file://{saved_path}"}
    model2 = LudwigModel(model2_config)
    train_stats2, preproc_data, _ = model2.train(
        dataset=new_training_set, output_directory=os.path.join(tmpdir, "model2")
    )
    # Assert that vocabulary of model2 input feature matches vocabulary of model1 encoder.
    model1_encoder_config = model1.config["input_features"][0]["encoder"]
    model2_encoder_config = model2.config["input_features"][0]["encoder"]
    assert model1_encoder_config["vocab"] == model2_encoder_config["vocab"]
