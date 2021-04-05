#! /usr/bin/env python
# coding=utf-8
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
import argparse
import logging
import os
import random
import string
import sys
import uuid
from typing import List

import numpy as np
import yaml

from ludwig.constants import (AUDIO, BAG, BINARY, CATEGORY, DATE, H3, IMAGE,
                              NUMERICAL, PREPROCESSING, SEQUENCE, SET,
                              TEXT, TIMESERIES, TYPE, VECTOR, NAME)
from ludwig.contrib import contrib_command, contrib_import
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import save_csv
from ludwig.utils.h3_util import components_to_h3
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)

letters = string.ascii_letters

DATETIME_FORMATS = {
    '%m-%d-%Y': '{m:02d}-{d:02d}-{Y:04d}',
    '%m-%d-%Y %H:%M:%S': '{m:02d}-{d:02d}-{Y:04d} {H:02d}:{M:02d}:{S:02d}',
    '%m/%d/%Y': '{m:02d}/{d:02d}/{Y:04d}',
    '%m/%d/%Y %H:%M:%S': '{m:02d}/{d:02d}/{Y:04d} {H:02d}:{M:02d}:{S:02d}',
    '%m-%d-%y': '{m:02d}-{d:02d}-{y:02d}',
    '%m-%d-%y %H:%M:%S': '{m:02d}-{d:02d}-{y:02d} {H:02d}:{M:02d}:{S:02d}',
    '%m/%d/%y': '{m:02d}/{d:02d}/{y:02d}',
    '%m/%d/%y %H:%M:%S': '{m:02d}/{d:02d}/{y:02d} {H:02d}:{M:02d}:{S:02d}',
    '%d-%m-%Y': '{d:02d}-{m:02d}-{Y:04d}',
    '%d-%m-%Y %H:%M:%S': '{d:02d}-{m:02d}-{Y:04d} {H:02d}:{M:02d}:{S:02d}',
    '%d/%m/%Y': '{d:02d}/{m:02d}/{Y:04d}',
    '%d/%m/%Y %H:%M:%S': '{d:02d}/{m:02d}/{Y:04d} {H:02d}:{M:02d}:{S:02d}',
    '%d-%m-%y': '{d:02d}-{m:02d}-{y:02d}',
    '%d-%m-%y %H:%M:%S': '{d:02d}-{m:02d}-{y:02d} {H:02d}:{M:02d}:{S:02d}',
    '%d/%m/%y': '{d:02d}/{m:02d}/{y:02d}',
    '%d/%m/%y %H:%M:%S': '{d:02d}/{m:02d}/{y:02d} {H:02d}:{M:02d}:{S:02d}',
    '%y-%m-%d': '{y:02d}-{m:02d}-{d:02d}',
    '%y-%m-%d %H:%M:%S': '{y:02d}-{m:02d}-{d:02d} {H:02d}:{M:02d}:{S:02d}',
    '%y/%m/%d': '{y:02d}/{m:02d}/{d:02d}',
    '%y/%m/%d %H:%M:%S': '{y:02d}/{m:02d}/{d:02d} {H:02d}:{M:02d}:{S:02d}',
    '%Y-%m-%d': '{Y:04d}-{m:02d}-{d:02d}',
    '%Y-%m-%d %H:%M:%S': '{Y:04d}-{m:02d}-{d:02d} {H:02d}:{M:02d}:{S:02d}',
    '%Y/%m/%d': '{Y:04d}/{m:02d}/{d:02d}',
    '%Y/%m/%d %H:%M:%S': '{Y:04d}/{m:02d}/{d:02d} {H:02d}:{M:02d}:{S:02d}',
    '%y-%d-%m': '{y:02d}-{d:02d}-{m:02d}',
    '%y-%d-%m %H:%M:%S': '{y:02d}-{d:02d}-{m:02d} {H:02d}:{M:02d}:{S:02d}',
    '%y/%d/%m': '{y:02d}/{d:02d}/{m:02d}',
    '%y/%d/%m %H:%M:%S': '{y:02d}/{d:02d}/{m:02d} {H:02d}:{M:02d}:{S:02d}',
    '%Y-%d-%m': '{Y:04d}-{d:02d}-{m:02d}',
    '%Y-%d-%m %H:%M:%S': '{Y:04d}-{d:02d}-{m:02d} {H:02d}:{M:02d}:{S:02d}',
    '%Y/%d/%m': '{Y:04d}/{d:02d}/{m:02d}',
    '%Y/%d/%m %H:%M:%S': '{Y:04d}/{d:02d}/{m:02d} {H:02d}:{M:02d}:{S:02d}'
}


def generate_string(length):
    sequence = []
    for _ in range(length):
        sequence.append(random.choice(letters))
    return ''.join(sequence)


def build_vocab(size):
    vocab = []
    for _ in range(size):
        vocab.append(generate_string(random.randint(2, 10)))
    return vocab


def return_none(feature):
    return None


def assign_vocab(feature):
    feature['idx2str'] = build_vocab(feature.get('vocab_size', 10))


def build_feature_parameters(features):
    feature_parameters = {}
    for feature in features:
        fearure_builder_function = get_from_registry(
            feature[TYPE],
            parameters_builders_registry
        )

        feature_parameters[feature[NAME]] = fearure_builder_function(feature)
    return feature_parameters


parameters_builders_registry = {
    'category': assign_vocab,
    'text': assign_vocab,
    'numerical': return_none,
    'binary': return_none,
    'set': assign_vocab,
    'bag': assign_vocab,
    'sequence': assign_vocab,
    'timeseries': return_none,
    'image': return_none,
    'audio': return_none,
    'date': return_none,
    'h3': return_none,
    VECTOR: return_none
}


def build_synthetic_dataset(dataset_size: int, features: List[dict]):
    """Symthesizes a dataset for testing purposes

    :param dataset_size: (int) size of the dataset
    :param features: (List[dict]) list of features to generate in YAML format.
        Provide a list contaning one dictionary for each feature,
        each dictionary must include a name, a type
        and can include some generation parameters depending on the type

    Example content for features:

    [
        {name: text_1, type: text, vocab_size: 20, max_len: 20},
        {name: text_2, type: text, vocab_size: 20, max_len: 20},
        {name: category_1, type: category, vocab_size: 10},
        {name: category_2, type: category, vocab_size: 15},
        {name: numerical_1, type: numerical},
        {name: numerical_2, type: numerical},
        {name: binary_1, type: binary},
        {name: binary_2, type: binary},
        {name: set_1, type: set, vocab_size: 20, max_len: 20},
        {name: set_2, type: set, vocab_size: 20, max_len: 20},
        {name: bag_1, type: bag, vocab_size: 20, max_len: 10},
        {name: bag_2, type: bag, vocab_size: 20, max_len: 10},
        {name: sequence_1, type: sequence, vocab_size: 20, max_len: 20},
        {name: sequence_2, type: sequence, vocab_size: 20, max_len: 20},
        {name: timeseries_1, type: timeseries, max_len: 20},
        {name: timeseries_2, type: timeseries, max_len: 20},
        {name: date_1, type: date},
        {name: date_2, type: date},
        {name: h3_1, type: h3},
        {name: h3_2, type: h3},
        {name: vector_1, type: vector},
        {name: vector_2, type: vector},
    ]
    """
    build_feature_parameters(features)
    header = []
    for feature in features:
        header.append(feature[NAME])

    yield header
    for _ in range(dataset_size):
        yield generate_datapoint(features)


def generate_datapoint(features):
    datapoint = []
    for feature in features:
        if ('cycle' in feature and feature['cycle'] is True and
                feature[TYPE] in cyclers_registry):
            cycler_function = cyclers_registry[feature[TYPE]]
            feature_value = cycler_function(feature)
        else:
            generator_function = get_from_registry(
                feature[TYPE],
                generators_registry
            )
            feature_value = generator_function(feature)
        datapoint.append(feature_value)
    return datapoint


def generate_category(feature):
    return random.choice(feature['idx2str'])


def generate_numerical(feature):
    return random.uniform(
        feature['min'] if 'min' in feature else 0,
        feature['max'] if 'max' in feature else 1
    )


def generate_binary(feature):
    p = feature['prob'] if 'prob' in feature else 0.5
    return np.random.choice([True, False], p=[p, 1 - p])


def generate_sequence(feature):
    length = feature.get('max_len', 10)
    if 'min_len' in feature:
        length = random.randint(feature['min_len'], length)
    sequence = [random.choice(feature['idx2str']) for _ in range(length)]
    return ' '.join(sequence)


def generate_set(feature):
    elems = []
    for _ in range(random.randint(0, feature.get('max_len', 3))):
        elems.append(random.choice(feature['idx2str']))
    return ' '.join(list(set(elems)))


def generate_bag(feature):
    elems = []
    for _ in range(random.randint(0, feature.get('max_len', 3))):
        elems.append(random.choice(feature['idx2str']))
    return ' '.join(elems)


def generate_text(feature):
    length = feature.get('max_len', 10)
    text = []
    for _ in range(random.randint(length - int(length * 0.2), length)):
        text.append(random.choice(feature['idx2str']))
    return ' '.join(text)


def generate_timeseries(feature):
    series = []
    for _ in range(feature.get('max_len', 10)):
        series.append(str(
            random.uniform(feature.get('min', 0), feature.get('max', 1))
        ))
    return ' '.join(series)


def generate_audio(feature):
    try:
        import soundfile
    except ImportError:
        logger.error(
            ' soundfile is not installed. '
            'In order to install all audio feature dependencies run '
            'pip install ludwig[audio]'
        )
        sys.exit(-1)

    destination_folder = feature.get('destination_folder', 'audio_files')
    if PREPROCESSING in feature:
        audio_length = feature[PREPROCESSING].get(
            'audio_file_length_limit_in_s', 2
        )
    else:
        audio_length = feature.get('audio_file_length_limit_in_s', 1)
    sampling_rate = 16000
    num_samples = int(audio_length * sampling_rate)
    audio = np.sin(np.arange(num_samples) / 100 * 2 * np.pi) * 2 * (
            np.random.random(num_samples) - 0.5)
    audio_filename = uuid.uuid4().hex[:10].upper() + '.wav'

    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        audio_dest_path = os.path.join(destination_folder, audio_filename)
        soundfile.write(audio_dest_path, audio, sampling_rate)

    except IOError as e:
        raise IOError(
            'Unable to create a folder for audio or save audio to disk.'
            '{0}'.format(e))

    return audio_dest_path


def generate_image(feature):
    try:
        from skimage.io import imsave
    except ImportError:
        logger.error(
            ' scikit-image is not installed. '
            'In order to install all image feature dependencies run '
            'pip install ludwig[image]'
        )
        sys.exit(-1)

    # Read num_channels, width, height
    destination_folder = feature.get('destination_folder', 'image_files')
    if PREPROCESSING in feature:
        height = feature[PREPROCESSING].get('height', 28)
        width = feature[PREPROCESSING].get('width', 28)
        num_channels = feature[PREPROCESSING].get('num_channels', 1)
    else:
        height = feature.get('height', 28)
        width = feature.get('width', 28)
        num_channels = feature.get('num_channels', 1)

    if width <= 0 or height <= 0 or num_channels < 1:
        raise ValueError('Invalid arguments for generating images')

    # Create a Random Image
    if num_channels == 1:
        img = np.random.rand(width, height) * 255
    else:
        img = np.random.rand(width, height, num_channels) * 255.0

    # Generate a unique random filename
    image_filename = uuid.uuid4().hex[:10].upper() + '.jpg'

    # Save the image to disk either in a specified location/new folder
    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        image_dest_path = os.path.join(destination_folder, image_filename)
        imsave(image_dest_path, img.astype('uint8'))

    except IOError as e:
        raise IOError(
            'Unable to create a folder for images/save image to disk.'
            '{0}'.format(e))

    return image_dest_path


def generate_datetime(feature):
    """picking a format among different types.
    If no format is specified, the first one is used.
    """
    if 'datetime_format' in feature:
        datetime_generation_format = DATETIME_FORMATS[
            feature['datetime_format']
        ]
    elif ('preprocessing' in feature and
          'datetime_format' in feature['preprocessing']):
        datetime_generation_format = DATETIME_FORMATS[
            feature['preprocessing']['datetime_format']
        ]
    else:
        datetime_generation_format = next(iter(DATETIME_FORMATS))

    y = random.randint(1, 99)
    Y = random.randint(1, 9999)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    H = random.randint(1, 12)
    M = random.randint(1, 59)
    S = random.randint(1, 59)

    return datetime_generation_format.format(y=y, Y=Y, m=m, d=d, H=H, M=M, S=S)


def generate_h3(feature):
    resolution = random.randint(0, 15)  # valid values [0, 15]
    h3_components = {
        'mode': 1,  # we can avoid testing other modes
        'edge': 0,  # only used in other modes
        'resolution': resolution,
        'base_cell': random.randint(0, 121),  # valid values [0, 121]
        # valid values [0, 7]
        'cells': [random.randint(0, 7) for _ in range(resolution)]
    }

    return components_to_h3(h3_components)


def generate_vector(feature):
    # Space delimited string with floating point numbers
    return ' '.join(
        [str(100 * random.random())
         for _ in range(feature.get('vector_size', 10))]
    )


generators_registry = {
    BINARY: generate_binary,
    NUMERICAL: generate_numerical,
    CATEGORY: generate_category,
    SET: generate_set,
    BAG: generate_bag,
    SEQUENCE: generate_sequence,
    TEXT: generate_sequence,
    TIMESERIES: generate_timeseries,
    IMAGE: generate_image,
    AUDIO: generate_audio,
    H3: generate_h3,
    DATE: generate_datetime,
    VECTOR: generate_vector

}

category_cycle = 0


def cycle_category(feature):
    global category_cycle
    if category_cycle >= len(feature['idx2str']):
        category_cycle = 0
    category = feature['idx2str'][category_cycle]
    category_cycle += 1
    return category


binary_cycle = False


def cycle_binary(feature):
    global binary_cycle
    if binary_cycle:
        binary_cycle = False
        return True
    else:
        binary_cycle = True
        return False


cyclers_registry = {
    'category': cycle_category,
    'binary': cycle_binary
}


def cli_synthesize_dataset(
        dataset_size: int,
        features: List[dict],
        output_path: str
) -> None:
    """Symthesizes a dataset for testing purposes

    :param dataset_size: (int) size of the dataset
    :param features: (List[dict]) list of features to generate in YAML format.
        Provide a list contaning one dictionary for each feature,
        each dictionary must include a name, a type
        and can include some generation parameters depending on the type
    :param output_path: (str) path where to save the output CSV file

    Example content for features:

    [
        {name: text_1, type: text, vocab_size: 20, max_len: 20},
        {name: text_2, type: text, vocab_size: 20, max_len: 20},
        {name: category_1, type: category, vocab_size: 10},
        {name: category_2, type: category, vocab_size: 15},
        {name: numerical_1, type: numerical},
        {name: numerical_2, type: numerical},
        {name: binary_1, type: binary},
        {name: binary_2, type: binary},
        {name: set_1, type: set, vocab_size: 20, max_len: 20},
        {name: set_2, type: set, vocab_size: 20, max_len: 20},
        {name: bag_1, type: bag, vocab_size: 20, max_len: 10},
        {name: bag_2, type: bag, vocab_size: 20, max_len: 10},
        {name: sequence_1, type: sequence, vocab_size: 20, max_len: 20},
        {name: sequence_2, type: sequence, vocab_size: 20, max_len: 20},
        {name: timeseries_1, type: timeseries, max_len: 20},
        {name: timeseries_2, type: timeseries, max_len: 20},
        {name: date_1, type: date},
        {name: date_2, type: date},
        {name: h3_1, type: h3},
        {name: h3_2, type: h3},
        {name: vector_1, type: vector},
        {name: vector_2, type: vector},
    ]
    """
    if dataset_size is None or features is None or output_path is None:
        raise ValueError(
            "Missing one or more required parameters: '--daset_size', "
            "'--features' or '--output_path'"
        )
    dataset = build_synthetic_dataset(dataset_size, features)
    save_csv(output_path, dataset)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script generates a synthetic dataset.',
        prog='ludwig synthesize_dataset',
        usage='%(prog)s [options]'
    )
    parser.add_argument(
        '-od',
        '--output_path',
        type=str,
        help='output CSV file path'
    )
    parser.add_argument(
        '-d',
        '--dataset_size',
        help='size of the dataset',
        type=int,
        default=100
    )
    parser.add_argument(
        '-f',
        '--features',
        default='[\
          {name: text_1, type: text, vocab_size: 20, max_len: 20}, \
          {name: text_2, type: text, vocab_size: 20, max_len: 20}, \
          {name: category_1, type: category, vocab_size: 10}, \
          {name: category_2, type: category, vocab_size: 15}, \
          {name: numerical_1, type: numerical}, \
          {name: numerical_2, type: numerical}, \
          {name: binary_1, type: binary}, \
          {name: binary_2, type: binary}, \
          {name: set_1, type: set, vocab_size: 20, max_len: 20}, \
          {name: set_2, type: set, vocab_size: 20, max_len: 20}, \
          {name: bag_1, type: bag, vocab_size: 20, max_len: 10}, \
          {name: bag_2, type: bag, vocab_size: 20, max_len: 10}, \
          {name: sequence_1, type: sequence, vocab_size: 20, max_len: 20}, \
          {name: sequence_2, type: sequence, vocab_size: 20, max_len: 20}, \
          {name: timeseries_1, type: timeseries, max_len: 20}, \
          {name: timeseries_2, type: timeseries, max_len: 20}, \
          {name: date_1, type: date}, \
          {name: date_2, type: date}, \
          {name: h3_1, type: h3}, \
          {name: h3_2, type: h3}, \
          {name: vector_1, type: vector}, \
          {name: vector_2, type: vector}, \
        ]',
        type=yaml.safe_load,
        help='list of features to generate in YAML format. '
             'Provide a list containing one dictionary for each feature, '
             'each dictionary must include a name, a type '
             'and can include some generation parameters depending on the type'
    )
    args = parser.parse_args(sys_argv)

    # No log level parameter this is placeholder if we add at later date
    # args.logging_level = logging_level_registry[args.logging_level]
    # logging.getLogger('ludwig').setLevel(
    #     args.logging_level
    # )
    # global logger
    # logger = logging.getLogger('ludwig.data.dataset_synthesizer')


    print_ludwig('Synthesize Dataset', LUDWIG_VERSION)

    cli_synthesize_dataset(**vars(args))


if __name__ == '__main__':
    contrib_import()
    contrib_command("synthesize_dataset", *sys.argv)
    cli(sys.argv[1:])
