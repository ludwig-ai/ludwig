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

import numpy as np
import pandas as pd
import pytest

from ludwig.experiment import experiment_cli
from tests.integration_tests.utils import generate_data, run_experiment, sequence_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


@pytest.mark.skip(reason="Issue #1333: Sequence output generation.")
@pytest.fixture(scope="module")
def generate_deterministic_sequence(num_records=200):
    in_vocab = [x + str(d) for x in list("abcde") for d in range(1, 4)]

    def generate_output(x):
        letter = x[0]
        repeat = int(x[1])
        return " ".join(repeat * letter)

    input_list = []
    output_list = []
    for _ in range(num_records):
        n = np.random.randint(1, 7, 1)
        input_seq = np.random.choice(in_vocab, n, replace=True)
        output_seq = [generate_output(x) for x in input_seq]
        input_list.append(" ".join(input_seq))
        output_list.append(" ".join(output_seq))

    train = {"in_seq": input_list, "out_seq": output_list}

    return pd.DataFrame(train)


# testing only a subset of options to reduce test run time
# combinations selected to test are the major tensor structures/sizes expected
# to be encountered: AttentionWrapperState, BeamSearchDecoderState, None
@pytest.mark.skip(reason="Issue #1333: Sequence output generation.")
@pytest.mark.parametrize("loss_sampler", ["learned_unigram", "fixed_unigram", "log_uniform", "uniform"])
@pytest.mark.parametrize("dec_attention", [None, "luong"])
@pytest.mark.parametrize("dec_cell_type", ["gru", "lstm"])
@pytest.mark.parametrize("enc_cell_type", ["rnn", "lstm"])
@pytest.mark.parametrize("enc_encoder", ["rnn"])
@pytest.mark.parametrize("dec_beam_width", [1, 2])
@pytest.mark.parametrize("dec_num_layers", [1, 2])
def test_sequence_generator(
    enc_encoder,
    enc_cell_type,
    dec_cell_type,
    dec_attention,
    dec_beam_width,
    dec_num_layers,
    loss_sampler,
    generate_deterministic_sequence,
):
    # Define input and output features
    input_features = [
        {
            "name": "in_seq",
            "type": "sequence",
            "encoder": enc_encoder,
            "cell_type": enc_cell_type,
            "reduce_output": None,
        }
    ]
    output_features = [
        {
            "name": "out_seq",
            "type": "sequence",
            "cell_type": dec_cell_type,
            "num_layers": dec_num_layers,
            "beam_width": dec_beam_width,
            "decoder": "generator",
            "attention": dec_attention,
            "reduce_input": None,
            "loss": {"type": "sampled_softmax_cross_entropy", "negative_samples": 10, "sampler": loss_sampler},
        }
    ]
    model_definition = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "fc_size": 14},  # 'concat'
        "training": {
            "" "epochs": 2,
            "early_stop": 5,
            "batch_size": 80,
            "optimizer": {"type": "adam"},
            "learning_rate": 0.001,
        },
    }
    args = {
        "config": model_definition,
        "skip_save_processed_input": True,
        "skip_save_progress": True,
        "skip_save_unprocessed_output": True,
        "skip_save_model": True,
        "skip_save_log": True,
        "debug": False,
    }
    # Generate test data
    np.random.seed(42)  # 13
    df = generate_deterministic_sequence

    # run the experiment
    results = experiment_cli(dataset=df, **args)


@pytest.mark.skip(reason="Issue #1333: Sequence output generation.")
@pytest.mark.parametrize("enc_cell_type", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("attention", [False, True])
def test_sequence_tagger(enc_cell_type, attention, csv_filename):
    # Define input and output features
    input_features = [sequence_feature(max_len=10, encoder="rnn", cell_type=enc_cell_type, reduce_output=None)]
    output_features = [
        sequence_feature(
            max_len=10,
            decoder="tagger",
            attention=attention,
            reduce_input=None,
        )
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # setup sampled softmax loss
    output_features[0].update({"loss": {"type": "sampled_softmax_cross_entropy", "negative_samples": 7}})

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path)
