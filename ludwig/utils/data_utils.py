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
import collections
import csv
import json
import logging
import os.path
import pickle
import random

import h5py
import numpy as np
import pandas as pd
from pandas.errors import ParserError


def load_csv(data_fp):
    data = []
    with open(data_fp, 'rb') as f:
        data = list(csv.reader(f))
    return data


def read_csv(data_fp, header=0):
    """
    Helper method to read a csv file. Wraps around pd.read_csv to handle some
    exceptions. Can extend to cover cases as necessary
    :param data_fp: path to the csv file
    :return: Pandas dataframe with the data
    """
    try:
        df = pd.read_csv(data_fp, header=header)
    except ParserError:
        logging.warning('Failed to parse the CSV with pandas default way,'
                        ' trying \\ as escape character.')
        df = pd.read_csv(data_fp, header=header, escapechar='\\')

    return df


def save_csv(data_fp, data):
    with open(data_fp, 'w', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            if not isinstance(row, collections.Iterable) or isinstance(row, str):
                row = [row]
            writer.writerow(row)


def load_json(data_fp):
    data = []
    with open(data_fp, 'r') as input_file:
        data = json.load(input_file)
    return data


def save_json(data_fp, data, sort_keys=True, indent=4):
    with open(data_fp, 'w') as output_file:
        json.dump(data, output_file, cls=NumpyEncoder, sort_keys=sort_keys,
                  indent=indent)


# to be tested
# also, when loading an hdf5 file
# most of the times you don't want
# to put everything in memory
# like this function does
# it's jsut for convenience for relatively small datasets
def load_hdf5(data_fp):
    data = {}
    with h5py.File(data_fp, 'r') as h5_file:
        for key in h5_file.keys():
            data[key] = h5_file[key].value
    return data


# def save_hdf5(data_fp: str, data: Dict[str, object]):
def save_hdf5(data_fp, data, metadata=None):

    if metadata is None:
        metadata = {}
    mode = 'w'
    if os.path.isfile(data_fp):
        mode = 'r+'
    with h5py.File(data_fp, mode) as h5_file:
        for key, value in data.items():
            dataset = h5_file.create_dataset(key, data=value)
            if key in metadata:
                if 'in_memory' in metadata[key]['preprocessing']:
                    if metadata[key]['preprocessing']['in_memory']:
                        dataset.attrs['in_memory'] = True
                    else:
                        dataset.attrs['in_memory'] = False


def load_object(object_fp):
    with open(object_fp, 'rb') as f:
        return pickle.load(f)


def save_object(object_fp, obj):
    with open(object_fp, 'wb') as f:
        pickle.dump(obj, f)


def load_array(data_fp, dtype=float):
    list_num = []
    with open(data_fp, 'r') as input_file:
        for x in input_file:
            list_num.append(dtype(x.strip()))
    return np.array(list_num)


def load_matrix(data_fp, dtype=float):
    list_num = []
    with open(data_fp, 'r') as input_file:
        for row in input_file:
            list_num.append([dtype(elem) for elem in row.strip().split()])
    return np.squeeze(np.array(list_num))


def save_array(data_fp, array):
    with open(data_fp, 'w') as output_file:
        for x in np.nditer(array):
            output_file.write(str(x) + '\n')


def load_pretrained_embeddings(embeddings_path, vocab):
    embeddings = load_glove(embeddings_path)

    # find out the size of the embeddings
    embeddings_size = len(next(iter(embeddings.values())))

    # calculate an average embedding, to use for initializing missing words
    avg_embedding = np.zeros(embeddings_size)
    count = 0
    for word in vocab:
        if word in embeddings:
            avg_embedding += embeddings[word]
            count += 1
    if count > 0:
        avg_embedding /= count

    # create the embedding matrix
    embeddings_vectors = []
    for word in vocab:
        if word in embeddings:
            embeddings_vectors.append(embeddings[word])
        else:
            embeddings_vectors.append(
                avg_embedding + np.random.uniform(-0.01, 0.01, embeddings_size))
    embeddings_matrix = np.stack(embeddings_vectors)

    # let's help the garbage collector free some memory
    embeddings = None

    return embeddings_matrix


def load_glove(file_path):
    logging.info('  Loading Glove format file {}'.format(file_path))
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line:
                split = line.split()
                word = split[0]
                embedding = np.array([float(val) for val in split[1:]])
                embeddings[word] = embedding
    logging.info('  {0} embeddings loaded'.format(len(embeddings)))
    return embeddings


def split_data(split, data):
    # type: (float, list) -> (list, list)
    split_length = int(round(split * len(data)))
    random.shuffle(data)
    return data[:split_length], data[split_length:]


def shuffle_unison_inplace(list_of_lists, random_state=None):
    if list_of_lists:
        assert all(len(l) == len(list_of_lists[0]) for l in list_of_lists)
        if random_state is not None:
            random_state.permutation(len(list_of_lists[0]))
        else:
            p = np.random.permutation(len(list_of_lists[0]))
        return [l[p] for l in list_of_lists]
    return None


def shuffle_dict_unison_inplace(np_dict, random_state=None):
    keys = list(np_dict.keys())
    list_of_lists = list(np_dict.values())

    # shuffle up the list of lists according to previous fct
    shuffled_list = shuffle_unison_inplace(list_of_lists, random_state)

    recon = {}
    for ii in range(len(keys)):
        dkey = keys[ii]
        recon[dkey] = shuffled_list[ii]

    # we've shuffled the dictionary in place!
    return recon


def shuffle_inplace(np_dict):
    if len(np_dict) == 0:
        return
    size = np_dict[next(iter(np_dict))].shape[0]
    for k in np_dict:
        if np_dict[k].shape[0] != size:
            raise ValueError(
                'Invalid: dictionary contains variable length arrays')

    p = np.random.permutation(size)

    for k in np_dict:
        np_dict[k] = np_dict[k][p]


def split_dataset_tvt(dataset, split):
    if 'split' in dataset:
        del dataset['split']
    training_set = split_dataset(dataset, split, value_to_split=0)
    validation_set = split_dataset(dataset, split, value_to_split=1)
    test_set = split_dataset(dataset, split, value_to_split=2)
    return training_set, test_set, validation_set


def split_dataset(dataset, split, value_to_split=0):
    splitted_dataset = {}
    for key in dataset:
        splitted_dataset[key] = dataset[key][split == value_to_split]
        if len(splitted_dataset[key]) == 0:
            return None
    return splitted_dataset


def collapse_rare_labels(labels, labels_limit):
    if labels_limit > 0:
        labels[labels >= labels_limit] = labels_limit
    return labels


def class_counts(dataset, labels_field):
    return np.bincount(dataset[labels_field].flatten()).tolist()


def text_feature_data_field(text_feature):
    return text_feature['name'] + '_' + text_feature['level']


def load_from_file(file_name, field=None, dtype=int):
    if file_name.endswith('.hdf5') and field is not None:
        hdf5_data = h5py.File(file_name, 'r')
        split = hdf5_data['split'].value
        column = hdf5_data[field].value
        hdf5_data.close()
        array = column[split == 2]  # ground truth
    elif file_name.endswith('.npy'):
        array = np.load(file_name)
    elif file_name.endswith('.csv'):
        array = read_csv(file_name, header=None)[0].tolist()
    else:
        array = load_matrix(file_name, dtype)
    return array


def replace_file_extension(file_path, desired_format):
    """
    Return a file path for a file with same name but different format.
    a.csv, json -> a.json
    a.csv, hdf5 -> a.hdf5
    :param file_path: original file path
    :param desired_format: desired file format
    :return: file path with same name but different format
    """
    if '.' in desired_format:
        # Handle the case if the user calls with '.hdf5' instead of 'hdf5'
        desired_format = desired_format.replace('.', '').strip()

    return os.path.splitext(file_path)[0] + '.' + desired_format


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)
