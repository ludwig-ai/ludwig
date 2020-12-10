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
import itertools
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.constants import BINARY, SEQUENCE, TEXT, SET
from ludwig.utils.neuropod_utils import export_neuropod
from ludwig.utils.strings_utils import str2bool
from tests.integration_tests.utils import category_feature, binary_feature, \
    numerical_feature, text_feature, set_feature, vector_feature, \
    image_feature, \
    audio_feature, timeseries_feature, date_feature, h3_feature, bag_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


def test_neuropod(csv_filename):
    #######
    # Setup
    #######
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = tmpdir
        data_csv_path = os.path.join(tmpdir, csv_filename)
        image_dest_folder = os.path.join(tmpdir, 'generated_images')
        audio_dest_folder = os.path.join(tmpdir, 'generated_audio')

        input_features = [
            binary_feature(),
            numerical_feature(),
            category_feature(vocab_size=3),
            sequence_feature(vocab_size=3),
            text_feature(vocab_size=3),
            vector_feature(),
            image_feature(image_dest_folder),
            audio_feature(audio_dest_folder),
            timeseries_feature(),
            date_feature(),
            h3_feature(),
            set_feature(vocab_size=3),
            bag_feature(vocab_size=3),
        ]

        output_features = [
            binary_feature(),
            numerical_feature(),
            category_feature(vocab_size=3),
            sequence_feature(vocab_size=3),
            text_feature(vocab_size=3),
            set_feature(vocab_size=3),
            vector_feature()
        ]

        # Generate test data
        data_csv_path = generate_data(input_features, output_features,
                                      data_csv_path)

        #############
        # Train model
        #############
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'training': {'epochs': 2}
        }
        ludwig_model = LudwigModel(config)
        ludwig_model.train(
            dataset=data_csv_path,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
            output_directory=dir_path
        )

        data_df = pd.read_csv(data_csv_path)
        original_predictions_df, _ = ludwig_model.predict(dataset=data_df)

        ###################
        # save Ludwig model
        ###################
        ludwigmodel_path = os.path.join(dir_path, 'ludwigmodel')
        shutil.rmtree(ludwigmodel_path, ignore_errors=True)
        ludwig_model.save(ludwigmodel_path)

        ################
        # build neuropod
        ################
        neuropod_path = os.path.join(dir_path, 'neuropod')
        shutil.rmtree(neuropod_path, ignore_errors=True)
        export_neuropod(ludwigmodel_path, neuropod_path=neuropod_path)

        ########################
        # predict using neuropod
        ########################
        if_dict = {
            input_feature['name']: np.expand_dims(np.array(
                [str(x) for x in data_df[input_feature['name']].tolist()],
                dtype='str'
            ), 1)
            for input_feature in input_features
        }

        from neuropod.loader import load_neuropod
        neuropod_model = load_neuropod(neuropod_path, _always_use_native=False)
        preds = neuropod_model.infer(if_dict)

        for key in preds:
            preds[key] = np.squeeze(preds[key])

        #########
        # cleanup
        #########
        # Delete the temporary data created
        for path in [ludwigmodel_path, neuropod_path,
                     image_dest_folder, audio_dest_folder]:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)

        ########
        # checks
        ########
        for output_feature in output_features:
            output_feature_name = output_feature['name']
            output_feature_type = output_feature['type']

            if (output_feature_name + "_predictions" in preds and
                    output_feature_name + "_predictions" in original_predictions_df):
                neuropod_pred = preds[
                    output_feature_name + "_predictions"].tolist()
                if output_feature_type == BINARY:
                    neuropod_pred = [ str2bool(x) for x in neuropod_pred]
                if output_feature_type in {SEQUENCE, TEXT, SET}:
                    neuropod_pred = [ x.split() for x in neuropod_pred]

                original_pred = original_predictions_df[
                    output_feature_name + "_predictions"].tolist()

                assert neuropod_pred == original_pred

            if (output_feature_name + "_probability" in preds and
                    output_feature_name + "_probability" in original_predictions_df):
                neuropod_prob = preds[
                    output_feature_name + "_probability"].tolist()
                if output_feature_type in {SEQUENCE, TEXT, SET}:
                    neuropod_prob = [ [float(n) for n in x.split()] for x in neuropod_prob]
                if any(isinstance(el, list) for el in neuropod_prob):
                    neuropod_prob = np.array(list(
                        itertools.zip_longest(*neuropod_prob, fillvalue=0)
                    )).T

                original_prob = original_predictions_df[
                    output_feature_name + "_probability"].tolist()
                if any(isinstance(el, list) for el in original_prob):
                    original_prob = np.array(list(
                        itertools.zip_longest(*original_prob, fillvalue=0)
                    )).T

                assert np.allclose(neuropod_prob, original_prob)

            if (output_feature_name + "_probabilities" in preds and
                    output_feature_name + "_probabilities" in original_predictions_df):
                neuropod_prob = preds[
                    output_feature_name + "_probabilities"].tolist()

                original_prob = original_predictions_df[
                    output_feature_name + "_probabilities"].tolist()

                assert np.allclose(neuropod_prob, original_prob)
