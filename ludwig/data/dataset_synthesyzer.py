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

import numpy as np
import yaml

from ludwig.constants import VECTOR
from ludwig.utils.data_utils import save_csv
from ludwig.utils.h3_util import components_to_h3
from ludwig.utils.misc import get_from_registry

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
    feature['idx2str'] = build_vocab(feature['vocab_size'])


def build_feature_parameters(features):
    feature_parameters = {}
    for feature in features:
        fearure_builder_function = get_from_registry(
            feature['type'],
            parameters_builders_registry
        )

        feature_parameters[feature['name']] = fearure_builder_function(feature)
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


def build_synthetic_dataset(dataset_size, features):
    build_feature_parameters(features)
    header = []
    for feature in features:
        header.append(feature['name'])

    yield header
    for _ in range(dataset_size):
        yield generate_datapoint(features)


def generate_datapoint(features):
    datapoint = []
    for feature in features:
        if ('cycle' in feature and feature['cycle'] is True and
                feature['type'] in cyclers_registry):
            cycler_function = cyclers_registry[feature['type']]
            feature_value = cycler_function(feature)
        else:
            generator_function = get_from_registry(
                feature['type'],
                generators_registry
            )
            feature_value = generator_function(feature)
        datapoint.append(feature_value)
    return datapoint


def generate_category(feature):
    return random.choice(feature['idx2str'])


def generate_text(feature):
    text = []
    for _ in range(random.randint(feature['max_len'] -
                                  int(feature['max_len'] * 0.2),
                                  feature['max_len'])):
        text.append(random.choice(feature['idx2str']))
    return ' '.join(text)


def generate_numerical(feature):
    return random.uniform(
        feature['min'] if 'min' in feature else 0,
        feature['max'] if 'max' in feature else 1
    )


def generate_binary(feature):
    p = feature['prob'] if 'prob' in feature else 0.5
    return np.random.choice([True, False], p=[p, 1 - p])


def generate_sequence(feature):
    length = feature['max_len']
    if 'min_len' in feature:
        length = random.randint(feature['min_len'], feature['max_len'])

    sequence = [random.choice(feature['idx2str']) for _ in range(length)]

    return ' '.join(sequence)


def generate_set(feature):
    elems = []
    for _ in range(random.randint(0, feature['max_len'])):
        elems.append(random.choice(feature['idx2str']))
    return ' '.join(list(set(elems)))


def generate_bag(feature):
    elems = []
    for _ in range(random.randint(0, feature['max_len'])):
        elems.append(random.choice(feature['idx2str']))
    return ' '.join(elems)


def generate_timeseries(feature):
    series = []
    for _ in range(feature['max_len']):
        series.append(
            str(
                random.uniform(
                    feature['min'] if 'min' in feature else 0,
                    feature['max'] if 'max' in feature else 1
                )
            )
        )
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

    audio_length = feature['preprocessing']['audio_file_length_limit_in_s']
    audio_dest_folder = feature['audio_dest_folder']
    sampling_rate = 16000
    num_samples = int(audio_length * sampling_rate)
    audio = np.sin(np.arange(num_samples) / 100 * 2 * np.pi) * 2 * (
            np.random.random(num_samples) - 0.5)
    audio_filename = uuid.uuid4().hex[:10].upper() + '.wav'

    try:
        if not os.path.exists(audio_dest_folder):
            os.makedirs(audio_dest_folder)

        audio_dest_path = os.path.join(audio_dest_folder, audio_filename)
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
    num_channels = feature['preprocessing']['num_channels']
    width = feature['preprocessing']['width']
    height = feature['preprocessing']['height']
    image_dest_folder = feature['destination_folder']

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
        if not os.path.exists(image_dest_folder):
            os.makedirs(image_dest_folder)

        image_dest_path = os.path.join(image_dest_folder, image_filename)
        imsave(image_dest_path, img.astype('uint8'))

    except IOError as e:
        raise IOError('Unable to create a folder for images/save image to disk.'
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
        datetime_generation_format = DATETIME_FORMATS[0]

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
        [str(100 * random.random()) for _ in range(feature['vector_size'])]
    )


generators_registry = {
    'category': generate_category,
    'text': generate_sequence,
    'numerical': generate_numerical,
    'binary': generate_binary,
    'set': generate_set,
    'bag': generate_bag,
    'sequence': generate_sequence,
    'timeseries': generate_timeseries,
    'image': generate_image,
    'audio': generate_audio,
    'h3': generate_h3,
    'date': generate_datetime,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script generates a synthetic dataset.')
    parser.add_argument('csv_file_path', help='output csv file path')
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
          ]',
        type=yaml.safe_load, help='dataset features'
    )
    args = parser.parse_args()

    dataset = build_synthetic_dataset(args.dataset_size, args.features)
    save_csv(args.csv_file_path, dataset)
