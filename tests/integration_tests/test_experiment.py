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
import os
import shutil
import uuid
from collections import namedtuple

import pytest
import yaml

from ludwig.api import LudwigModel
from ludwig.backend import LOCAL_BACKEND, LocalBackend
from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.experiment import experiment_cli
from ludwig.features.h3_feature import H3InputFeature
from ludwig.predict import predict_cli
from ludwig.utils.data_utils import read_csv
from ludwig.utils.defaults import default_random_seed

from tests.conftest import delete_temporary_data
from tests.integration_tests.utils import ENCODERS, HF_ENCODERS, \
    HF_ENCODERS_SHORT, slow, create_data_set_to_use
from tests.integration_tests.utils import audio_feature
from tests.integration_tests.utils import bag_feature
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import date_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import \
    generate_output_features_with_dependencies
from tests.integration_tests.utils import h3_feature
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import run_experiment
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import set_feature
from tests.integration_tests.utils import spawn
from tests.integration_tests.utils import text_feature
from tests.integration_tests.utils import timeseries_feature
from tests.integration_tests.utils import vector_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


class LocalTestBackend(LocalBackend):
    @property
    def supports_multiprocessing(self):
        return False


@pytest.mark.parametrize('encoder', ENCODERS)
def test_experiment_text_feature_non_HF(encoder, csv_filename):
    input_features = [
        text_feature(
            vocab_size=30,
            min_len=1,
            encoder=encoder,
            preprocessing={'word_tokenizer': 'space'}
        )
    ]
    output_features = [category_feature(vocab_size=2)]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


@spawn
def run_experiment_with_encoder(encoder, csv_filename):
    # Run in a subprocess to clear TF and prevent OOM
    # This also allows us to use GPU resources
    input_features = [
        text_feature(
            vocab_size=30,
            min_len=1,
            encoder=encoder,
        )
    ]
    output_features = [category_feature(vocab_size=2)]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize('encoder', HF_ENCODERS_SHORT)
def test_experiment_text_feature_HF(encoder, csv_filename):
    run_experiment_with_encoder(encoder, csv_filename)


@slow
@pytest.mark.parametrize('encoder', HF_ENCODERS)
def test_experiment_text_feature_HF_full(encoder, csv_filename):
    run_experiment_with_encoder(encoder, csv_filename)


def test_experiment_seq_seq(csv_filename):
    # Single Sequence input, single sequence output
    # Only the following encoders are working
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    encoders2 = ['cnnrnn', 'stacked_cnn']
    for encoder in encoders2:
        logger.info('seq to seq test, Encoder: {0}'.format(encoder))
        input_features[0]['encoder'] = encoder
        run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_seq_seq_model_def_file(csv_filename, yaml_filename):
    # seq-to-seq test to use config file instead of dictionary
    input_features = [text_feature(reduce_output=None, encoder='embed')]
    output_features = [
        text_feature(reduce_input=None, vocab_size=3, decoder='tagger')
    ]

    # Save the config to a yaml file
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }
    with open(yaml_filename, 'w') as yaml_out:
        yaml.safe_dump(config, yaml_out)

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(
        None, None, dataset=rel_path, config_file=yaml_filename
    )


def test_experiment_seq_seq_train_test_valid(csv_filename):
    # seq-to-seq test to use train, test, validation files
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [
        text_feature(reduce_input=None, vocab_size=3, decoder='tagger')
    ]

    train_csv = generate_data(
        input_features, output_features, 'tr_' + csv_filename
    )
    test_csv = generate_data(
        input_features, output_features, 'test_' + csv_filename, 20
    )
    valdation_csv = generate_data(
        input_features, output_features, 'val_' + csv_filename, 20
    )

    run_experiment(
        input_features,
        output_features,
        training_set=train_csv,
        test_set=test_csv,
        validation_set=valdation_csv
    )

    input_features[0]['encoder'] = 'parallel_cnn'
    # Save intermediate output
    run_experiment(
        input_features,
        output_features,
        training_set=train_csv,
        test_set=test_csv,
        validation_set=valdation_csv
    )

    # Delete the temporary data created
    # This test is saving the processed data to hdf5
    for prefix in ['tr_', 'test_', 'val_']:
        if os.path.isfile(prefix + csv_filename):
            os.remove(prefix + csv_filename)


def test_experiment_multi_input_intent_classification(csv_filename):
    # Multiple inputs, Single category output
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    for encoder in ENCODERS:
        input_features[0]['encoder'] = encoder
        run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_multiclass_with_class_weights(csv_filename):
    # Multiple inputs, Single category output
    input_features = [category_feature(vocab_size=10)]
    output_features = [category_feature(vocab_size=3,
                                        loss={"class_weights": [0, 1, 2, 3]})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_multilabel_with_class_weights(csv_filename):
    # Multiple inputs, Single category output
    input_features = [category_feature(vocab_size=10)]
    output_features = [set_feature(vocab_size=3,
                                   loss={"class_weights": [0, 0, 1, 2, 3]}
                                   )]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize(
    'output_features',
    [
        # baseline test case
        [
            category_feature(vocab_size=2, reduce_input='sum'),
            sequence_feature(vocab_size=10, max_len=5),
            numerical_feature()
        ],

        # use generator as decoder
        [
            category_feature(vocab_size=2, reduce_input='sum'),
            sequence_feature(vocab_size=10, max_len=5, decoder='generator'),
            numerical_feature()
        ],

        # Generator decoder and reduce_input = None
        [
            category_feature(vocab_size=2, reduce_input='sum'),
            sequence_feature(max_len=5, decoder='generator',
                             reduce_input=None),
            numerical_feature(normalization='minmax')
        ],

        # output features with dependencies single dependency
        generate_output_features_with_dependencies('feat3', ['feat1']),

        # output features with dependencies multiple dependencies
        generate_output_features_with_dependencies('feat3',
                                                   ['feat1', 'feat2']),

        # output features with dependencies multiple dependencies
        generate_output_features_with_dependencies('feat2',
                                                   ['feat1', 'feat3']),

        # output features with dependencies
        generate_output_features_with_dependencies('feat1', ['feat2'])
    ]
)
def test_experiment_multiple_seq_seq(csv_filename, output_features):
    input_features = [
        text_feature(vocab_size=100, min_len=1, encoder='stacked_cnn'),
        numerical_feature(normalization='zscore'),
        category_feature(vocab_size=10, embedding_size=5),
        set_feature(),
        sequence_feature(vocab_size=10, max_len=10, encoder='embed')
    ]
    output_features = output_features

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


ImageParms = namedtuple(
    'ImageTestParms',
    'image_encoder in_memory_flag skip_save_processed_input'
)


@pytest.mark.parametrize(
    'image_parms',
    [
        ImageParms('resnet', True, True),
        ImageParms('stacked_cnn', True, True),
        ImageParms('stacked_cnn', False, False)
    ]
)
def test_experiment_image_inputs(image_parms: ImageParms, csv_filename: str):
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='resnet',
            preprocessing={
                'in_memory': True,
                'height': 12,
                'width': 12,
                'num_channels': 3,
                'num_processes': 5
            },
            fc_size=16,
            num_filters=8
        ),
        text_feature(encoder='embed', min_len=1),
        numerical_feature(normalization='zscore')
    ]
    output_features = [
        category_feature(vocab_size=2, reduce_input='sum'),
        numerical_feature()
    ]

    input_features[0]['encoder'] = image_parms.image_encoder
    input_features[0]['preprocessing'][
        'in_memory'] = image_parms.in_memory_flag
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(
        input_features,
        output_features,
        dataset=rel_path,
        skip_save_processed_input=image_parms.skip_save_processed_input
    )

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)


IMAGE_DATA_FORMATS_TO_TEST = ['csv', 'df', 'hdf5']


@pytest.mark.parametrize('test_in_memory', [True, False])
@pytest.mark.parametrize('test_format', IMAGE_DATA_FORMATS_TO_TEST)
@pytest.mark.parametrize('train_in_memory', [True, False])
@pytest.mark.parametrize('train_format', IMAGE_DATA_FORMATS_TO_TEST)
def test_experiment_image_dataset(
        train_format, train_in_memory,
        test_format, test_in_memory
):
    # primary focus of this test is to determine if exceptions are
    # raised for different data set formats and in_memory setting
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='stacked_cnn',
            preprocessing={
                'in_memory': True,
                'height': 12,
                'width': 12,
                'num_channels': 3,
                'num_processes': 5
            },
            fc_size=16,
            num_filters=8
        ),
    ]
    output_features = [
        category_feature(vocab_size=2, reduce_input='sum'),
    ]

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {
            'type': 'concat',
            'fc_size': 14
        },
        'preprocessing': {},
        'training': {'epochs': 2}
    }

    # create temporary name for train and test data sets
    train_csv_filename = 'train_' + uuid.uuid4().hex[:10].upper() + '.csv'
    test_csv_filename = 'test_' + uuid.uuid4().hex[:10].upper() + '.csv'

    # setup training data format to test
    train_data = generate_data(input_features, output_features,
                               train_csv_filename)
    config['input_features'][0]['preprocessing']['in_memory'] \
        = train_in_memory
    training_set_metadata = None

    backend = LocalTestBackend()
    if train_format == 'hdf5':
        # hdf5 format
        train_set, _, _, training_set_metadata = preprocess_for_training(
            config,
            dataset=train_data,
            backend=backend,
        )
        train_dataset_to_use = train_set.data_hdf5_fp
    else:
        train_dataset_to_use = create_data_set_to_use(train_format, train_data)

    # define Ludwig model
    model = LudwigModel(
        config=config,
        backend=backend,
    )
    model.train(
        dataset=train_dataset_to_use,
        training_set_metadata=training_set_metadata
    )

    model.config['input_features'][0]['preprocessing']['in_memory'] \
        = test_in_memory

    # setup test data format to test
    test_data = generate_data(input_features, output_features,
                              test_csv_filename)

    if test_format == 'hdf5':
        # hdf5 format
        # create hdf5 data set
        _, test_set, _, training_set_metadata_for_test = preprocess_for_training(
            model.config,
            dataset=test_data,
            backend=backend,
        )
        test_dataset_to_use = test_set.data_hdf5_fp
    else:
        test_dataset_to_use = create_data_set_to_use(test_format, test_data)

    # run functions with the specified data format
    model.evaluate(dataset=test_dataset_to_use)
    model.predict(dataset=test_dataset_to_use)

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)
    delete_temporary_data(train_csv_filename)
    delete_temporary_data(test_csv_filename)


DATA_FORMATS_TO_TEST = [
    'csv', 'df', 'dict', 'excel', 'excel_xls', 'feather', 'fwf', 'hdf5', 'html',
    'json', 'jsonl', 'parquet', 'pickle', 'stata', 'tsv'
]
@pytest.mark.parametrize('data_format', DATA_FORMATS_TO_TEST)
def test_experiment_dataset_formats(data_format):
    # primary focus of this test is to determine if exceptions are
    # raised for different data set formats and in_memory setting

    input_features = [
        numerical_feature(),
        category_feature()
    ]
    output_features = [
        category_feature(),
        numerical_feature()
    ]

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {
            'type': 'concat',
            'fc_size': 14
        },
        'preprocessing': {},
        'training': {'epochs': 2}
    }

    # create temporary name for train and test data sets
    csv_filename = 'train_' + uuid.uuid4().hex[:10].upper() + '.csv'

    # setup training data format to test
    raw_data = generate_data(input_features, output_features,
                               csv_filename)

    training_set_metadata = None

    if data_format == 'hdf5':
        # hdf5 format
        training_set, _, _, training_set_metadata = preprocess_for_training(
            config,
            dataset=raw_data
        )
        dataset_to_use = training_set.data_hdf5_fp
    else:
        dataset_to_use = create_data_set_to_use(data_format, raw_data)

    # define Ludwig model
    model = LudwigModel(
        config=config
    )
    model.train(
        dataset=dataset_to_use,
        training_set_metadata=training_set_metadata,
        random_seed=default_random_seed
    )

    # # run functions with the specified data format
    model.evaluate(dataset=dataset_to_use)
    model.predict(dataset=dataset_to_use)

    # Delete the temporary data created
    delete_temporary_data(csv_filename)



def test_experiment_audio_inputs(csv_filename):
    # Audio Inputs
    audio_dest_folder = os.path.join(os.getcwd(), 'generated_audio')

    input_features = [
        audio_feature(
            folder=audio_dest_folder
        )
    ]
    output_features = [
        binary_feature()
    ]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)

    # Delete the temporary data created
    shutil.rmtree(audio_dest_folder)


def test_experiment_tied_weights(csv_filename):
    # Single sequence input, single category output
    input_features = [
        text_feature(
            name='text_feature1',
            min_len=1,
            encoder='cnnrnn',
            reduce_output='sum'
        ),
        text_feature(
            name='text_feature2',
            min_len=1,
            encoder='cnnrnn',
            reduce_output='sum',
            tied_weights='text_feature1'
        )
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    for encoder in ENCODERS:
        input_features[0]['encoder'] = encoder
        input_features[1]['encoder'] = encoder
        run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize('enc_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize('attention', [False, True])
def test_sequence_tagger(
        enc_cell_type,
        attention,
        csv_filename
):
    # Define input and output features
    input_features = [
        sequence_feature(
            max_len=10,
            encoder='rnn',
            cell_type=enc_cell_type,
            reduce_output=None
        )
    ]
    output_features = [
        sequence_feature(
            max_len=10,
            decoder='tagger',
            attention=attention,
            reduce_input=None
        )
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path)


def test_sequence_tagger_text(
        csv_filename
):
    # Define input and output features
    input_features = [
        text_feature(
            max_len=10,
            encoder='rnn',
            reduce_output=None
        )
    ]
    output_features = [
        sequence_feature(
            max_len=10,
            decoder='tagger',
            reduce_input=None
        )
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize('sequence_combiner_encoder', ENCODERS[:-2])
def test_experiment_sequence_combiner(sequence_combiner_encoder, csv_filename):
    # Sequence combiner
    input_features = [
        sequence_feature(
            name='seq1',
            min_len=5,
            max_len=5,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        ),
        sequence_feature(
            name='seq2',
            min_len=5,
            max_len=5,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        ),
        category_feature(vocab_size=5)
    ]
    output_features = [
        category_feature(reduce_input='sum', vocab_size=5)
    ]

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {
            'epochs': 2
        },
        'combiner': {
            'type': 'sequence',
            'encoder': 'rnn',
            'main_sequence_feature': 'seq1',
            'reduce_output': None,
        }
    }

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    for encoder in ENCODERS[:-2]:
        logger.error('sequence combiner. encoders: {0}, {1}'.format(
            encoder,
            encoder
        ))
        input_features[0]['encoder'] = encoder
        input_features[1]['encoder'] = encoder

        config['input_features'] = input_features

        exp_dir_name = experiment_cli(
            config,
            skip_save_processed_input=False,
            skip_save_progress=True,
            skip_save_unprocessed_output=True,
            dataset=rel_path
        )
        shutil.rmtree(exp_dir_name, ignore_errors=True)


def test_experiment_model_resume(csv_filename):
    # Single sequence input, single category output
    # Tests saving a model file, loading it to rerun training and predict
    input_features = [sequence_feature(encoder='rnn', reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    _, _, _, _, output_dir = experiment_cli(config, dataset=rel_path)
    logger.info('Experiment Directory: {0}'.format(output_dir))

    experiment_cli(
        config,
        dataset=rel_path,
        model_resume_path=output_dir
    )

    predict_cli(os.path.join(output_dir, 'model'), dataset=rel_path)
    shutil.rmtree(output_dir, ignore_errors=True)


def test_experiment_various_feature_types(csv_filename):
    input_features = [binary_feature(), bag_feature()]
    output_features = [set_feature(max_len=3, vocab_size=5)]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_timeseries(csv_filename):
    input_features = [timeseries_feature()]
    output_features = [binary_feature()]

    encoders2 = [
        'transformer'
    ]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    for encoder in encoders2:
        input_features[0]['encoder'] = encoder
        run_experiment(input_features, output_features, dataset=rel_path)


def test_visual_question_answering(csv_filename):
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='resnet',
            preprocessing={
                'in_memory': True,
                'height': 8,
                'width': 8,
                'num_channels': 3,
                'num_processes': 5
            },
            fc_size=8,
            num_filters=8
        ),
        text_feature(encoder='embed', min_len=1, level='word'),
    ]
    output_features = [sequence_feature(decoder='generator', cell_type='lstm')]
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)


def test_image_resizing_num_channel_handling(csv_filename):
    """
    This test creates two image datasets with 3 channels and 1 channel. The
    combination of this data is used to train a model. This checks the cases
    where the user may or may not specify a number of channels in the
    config
    :param csv_filename:
    :return:
    """
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='resnet',
            preprocessing={
                'in_memory': True,
                'height': 8,
                'width': 8,
                'num_channels': 3,
                'num_processes': 5
            },
            fc_size=8,
            num_filters=8
        ),
        text_feature(encoder='embed', min_len=1),
        numerical_feature(normalization='minmax')
    ]
    output_features = [binary_feature(), numerical_feature()]
    rel_path = generate_data(
        input_features, output_features, csv_filename, num_examples=50
    )

    df1 = read_csv(rel_path)

    input_features[0]['preprocessing']['num_channels'] = 1
    rel_path = generate_data(
        input_features, output_features, csv_filename, num_examples=50
    )
    df2 = read_csv(rel_path)

    df = concatenate_df(df1, df2, None, LOCAL_BACKEND)
    df.to_csv(rel_path, index=False)

    # Here the user sepcifiies number of channels. Exception shouldn't be thrown
    run_experiment(input_features, output_features, dataset=rel_path)

    del input_features[0]['preprocessing']['num_channels']

    # User now doesn't specify num channels. Should throw exception
    with pytest.raises(ValueError):
        run_experiment(input_features, output_features, dataset=rel_path)

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)


@pytest.mark.parametrize('encoder', ['wave', 'embed'])
def test_experiment_date(encoder, csv_filename):
    input_features = [date_feature()]
    output_features = [category_feature(vocab_size=2)]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    input_features[0]['encoder'] = encoder
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize('encoder', H3InputFeature.encoder_registry.keys())
def test_experiment_h3(encoder, csv_filename):
    input_features = [h3_feature()]
    output_features = [binary_feature()]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    input_features[0]['encoder'] = encoder
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_vector_feature_1(csv_filename):
    input_features = [vector_feature()]
    output_features = [binary_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_vector_feature_2(csv_filename):
    input_features = [vector_feature()]
    output_features = [vector_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_sampled_softmax(csv_filename):
    # Multiple inputs, Single category output
    input_features = [text_feature(vocab_size=10, min_len=1)]
    output_features = [category_feature(
        vocab_size=500,
        loss={'type': 'sampled_softmax_cross_entropy'}
    )]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename,
                             num_examples=10000)

    run_experiment(input_features, output_features, dataset=rel_path)


if __name__ == '__main__':
    """
    To run tests individually, run:
    ```python -m pytest tests/integration_tests/test_experiment.py::test_name```
    """
    pass
