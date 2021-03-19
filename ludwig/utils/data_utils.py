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
import functools
import json
import logging
import os.path
import pickle
import random
import re

import h5py
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from sklearn.model_selection import KFold

try:
    import dask.dataframe as dd
    DASK_DF_FORMATS = {dd.core.DataFrame}
except ImportError:
    DASK_DF_FORMATS = set()

from ludwig.constants import PREPROCESSING, SPLIT, PROC_COLUMN
from ludwig.globals import (MODEL_HYPERPARAMETERS_FILE_NAME,
                            MODEL_WEIGHTS_FILE_NAME,
                            TRAIN_SET_METADATA_FILE_NAME)

logger = logging.getLogger(__name__)

DATASET_SPLIT_URL = 'dataset_{}_fp'
DATA_PROCESSED_CACHE_DIR = 'data_processed_cache_dir'
DATA_TRAIN_HDF5_FP = 'data_train_hdf5_fp'
HDF5_COLUMNS_KEY = 'columns'
DICT_FORMATS = {'dict', 'dictionary', dict}
DATAFRAME_FORMATS = {'dataframe', 'df', pd.DataFrame} | DASK_DF_FORMATS
CSV_FORMATS = {'csv'}
TSV_FORMATS = {'tsv'}
JSON_FORMATS = {'json'}
JSONL_FORMATS = {'jsonl'}
EXCEL_FORMATS = {'excel'}
PARQUET_FORMATS = {'parquet'}
PICKLE_FORMATS = {'pickle'}
FEATHER_FORMATS = {'feather'}
FWF_FORMATS = {'fwf'}
HTML_FORMATS = {'html'}
ORC_FORMATS = {'orc'}
SAS_FORMATS = {'sas'}
SPSS_FORMATS = {'spss'}
STATA_FORMATS = {'stata'}
HDF5_FORMATS = {'hdf5', 'h5'}
CACHEABLE_FORMATS = set.union(*(CSV_FORMATS, TSV_FORMATS,
                                JSON_FORMATS, JSONL_FORMATS,
                                EXCEL_FORMATS, PARQUET_FORMATS, PICKLE_FORMATS,
                                FEATHER_FORMATS, FWF_FORMATS, HTML_FORMATS,
                                ORC_FORMATS, SAS_FORMATS, SPSS_FORMATS,
                                STATA_FORMATS))

PANDAS_DF = pd


def get_split_path(dataset_fp):
    return os.path.splitext(dataset_fp)[0] + '.split.csv'


def get_abs_path(data_csv_path, file_path):
    if data_csv_path is not None:
        return os.path.join(data_csv_path, file_path)
    else:
        return file_path


def load_csv(data_fp):
    data = []
    with open(data_fp, 'rb') as f:
        data = list(csv.reader(f))
    return data


def read_xsv(data_fp, df_lib=PANDAS_DF, separator=',', header=0, nrows=None, skiprows=None):
    """
    Helper method to read a csv file. Wraps around pd.read_csv to handle some
    exceptions. Can extend to cover cases as necessary
    :param data_fp: path to the xsv file
    :param df_lib: DataFrame library used to read in the CSV
    :param separator: defaults separator to use for splitting
    :param header: header argument for pandas to read the csv
    :param nrows: number of rows to read from the csv, None means all
    :param skiprows: number of rows to skip from the csv, None means no skips
    :return: Pandas dataframe with the data
    """
    with open(data_fp, 'r', encoding="utf8") as csvfile:
        try:
            dialect = csv.Sniffer().sniff(csvfile.read(1024 * 100),
                                          delimiters=[',', '\t', '|'])
            separator = dialect.delimiter
        except csv.Error:
            # Could not conclude the delimiter, defaulting to user provided
            pass

    try:
        df = df_lib.read_csv(data_fp, sep=separator, header=header,
                             nrows=nrows, skiprows=skiprows)
    except ParserError:
        logger.warning('Failed to parse the CSV with pandas default way,'
                       ' trying \\ as escape character.')
        df = df_lib.read_csv(data_fp, sep=separator, header=header,
                             escapechar='\\',
                             nrows=nrows, skiprows=skiprows)

    return df


read_csv = functools.partial(read_xsv, separator=',')
read_tsv = functools.partial(read_xsv, separator='\t')


def read_json(data_fp, df_lib, normalize=False):
    if normalize:
        return df_lib.json_normalize(load_json(data_fp))
    else:
        return df_lib.read_json(data_fp)


def read_jsonl(data_fp, df_lib):
    return df_lib.read_json(data_fp, lines=True)


def read_excel(data_fp, df_lib):
    fp_split = os.path.splitext(data_fp)
    if fp_split[1] == '.xls':
        excel_engine = 'xlrd'
    else:
        excel_engine = 'openpyxl'
    return df_lib.read_excel(data_fp, engine=excel_engine)


def read_parquet(data_fp, df_lib):
    return df_lib.read_parquet(data_fp)


def read_pickle(data_fp, df_lib):
    return df_lib.read_pickle(data_fp)


def read_fwf(data_fp, df_lib):
    return df_lib.read_fwf(data_fp)


def read_feather(data_fp, df_lib):
    return df_lib.read_feather(data_fp)


def read_html(data_fp, df_lib):
    return df_lib.read_html(data_fp)[0]


def read_orc(data_fp, df_lib):
    return df_lib.read_orc(data_fp)


def read_sas(data_fp, df_lib):
    return df_lib.read_sas(data_fp)


def read_spss(data_fp, df_lib):
    return df_lib.read_spss(data_fp)


def read_stata(data_fp, df_lib):
    return df_lib.read_stata(data_fp)


def save_csv(data_fp, data):
    with open(data_fp, 'w', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            if not isinstance(row, collections.Iterable) or isinstance(row,
                                                                       str):
                row = [row]
            writer.writerow(row)


def csv_contains_column(data_fp, column_name):
    return column_name in read_csv(data_fp, nrows=0)  # only loads header


def load_json(data_fp):
    with open(data_fp, 'r') as input_file:
        data = json.load(input_file)
    return data


def save_json(data_fp, data, sort_keys=True, indent=4):
    with open(data_fp, 'w') as output_file:
        json.dump(data, output_file, cls=NumpyEncoder, sort_keys=sort_keys,
                  indent=indent)


def to_numpy_dataset(df):
    dataset = {}
    for col in df.columns:
        dataset[col] = np.stack(df[col].to_numpy())
    return dataset


def from_numpy_dataset(dataset):
    col_mapping = {}
    for k, v in dataset.items():
        if len(v.shape) > 1:
            # unstacking, needed for ndarrays of dimension 2 and more
            *vals, = v
        else:
            # not unstacking. Needed because otherwise pandas casts types
            # the way it wants, like converting a list of float32 scalats
            # to a column of float64
            vals = v
        col_mapping[k] = vals
    return pd.DataFrame.from_dict(col_mapping)


def save_hdf5(data_fp, data):
    mode = 'w'
    if os.path.isfile(data_fp):
        mode = 'r+'

    numpy_dataset = to_numpy_dataset(data)
    with h5py.File(data_fp, mode) as h5_file:
        h5_file.create_dataset(HDF5_COLUMNS_KEY, data=np.array(data.columns.values, dtype='S'))
        for column in data.columns:
            h5_file.create_dataset(column, data=numpy_dataset[column])


def load_hdf5(data_fp):
    hdf5_data = h5py.File(data_fp, 'r')
    columns = [s.decode('utf-8') for s in hdf5_data[HDF5_COLUMNS_KEY][()].tolist()]

    numpy_dataset = {}
    for column in columns:
        numpy_dataset[column] = hdf5_data[column][()]

    return from_numpy_dataset(numpy_dataset)


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
                avg_embedding + np.random.uniform(-0.01, 0.01, embeddings_size)
            )
    embeddings_matrix = np.stack(embeddings_vectors)

    # let's help the garbage collector free some memory
    embeddings = None

    return embeddings_matrix


@functools.lru_cache(1)
def load_glove(file_path):
    logger.info('  Loading Glove format file {}'.format(file_path))
    embeddings = {}
    embedding_size = 0

    # collect embeddings size assuming the first line is correct
    with open(file_path, 'r', encoding='utf-8') as f:
        found_line = False
        while not found_line:
            line = f.readline()
            if line:
                embedding_size = len(line.split()) - 1
                found_line = True

    # collect embeddings
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            if line:
                try:
                    split = line.split()
                    if len(split) != embedding_size + 1:
                        raise ValueError
                    word = split[0]
                    embedding = np.array(
                        [float(val) for val in split[-embedding_size:]]
                    )
                    embeddings[word] = embedding
                except ValueError:
                    logger.warning(
                        'Line {} in the GloVe file {} is malformed, '
                        'skipping it'.format(
                            line_number, file_path
                        )
                    )
    logger.info('  {0} embeddings loaded'.format(len(embeddings)))
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
            p = random_state.permutation(len(list_of_lists[0]))
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


def split_dataset_ttv(dataset, split):
    training_set = split_dataset(dataset, split, 0)
    validation_set = split_dataset(dataset, split, 1)
    test_set = split_dataset(dataset, split, 2)
    return training_set, test_set, validation_set


def split_dataset(dataset, split, value_to_split=0):
    split_df = dataset[dataset[split] == value_to_split]
    if len(split_df) == 0:
        return None
    return split_df.reset_index()


def collapse_rare_labels(labels, labels_limit):
    if labels_limit > 0:
        labels[labels >= labels_limit] = labels_limit
    return labels


def class_counts(dataset, labels_field):
    return np.bincount(dataset[labels_field].flatten()).tolist()


def text_feature_data_field(text_feature):
    return text_feature[PROC_COLUMN] + '_' + text_feature['level']


def load_from_file(file_name, field=None, dtype=int, ground_truth_split=2):
    """Load experiment data from supported file formats.

    Experiment data can be test/train statistics, model predictions,
    probability, ground truth,  ground truth metadata.
    :param file_name: Path to file to be loaded
    :param field: Target Prediction field.
    :param dtype:
    :param ground_truth_split: Ground truth split filter where 0 is train 1 is
    validation and 2 is test split. By default test split is used when loading
    ground truth from hdf5.
    :return: Experiment data as array
    """
    if file_name.endswith('.hdf5') and field is not None:
        dataset = pd.read_hdf(file_name, key=HDF5_COLUMNS_KEY)
        column = dataset[field]
        array = column[dataset[SPLIT] == ground_truth_split].values  # ground truth
    elif file_name.endswith('.npy'):
        array = np.load(file_name)
    elif file_name.endswith('.csv'):
        array = read_csv(file_name, header=None).values
    else:
        array = load_matrix(file_name, dtype)
    return array


def replace_file_extension(file_path, extension):
    """
    Return a file path for a file with same name but different format.
    a.csv, json -> a.json
    a.csv, hdf5 -> a.hdf5
    :param file_path: original file path
    :param extension: file extension
    :return: file path with same name but different format
    """
    if file_path is None:
        return None
    extension = extension.strip()
    if extension.startswith('.'):
        # Handle the case if the user calls with '.hdf5' instead of 'hdf5'
        extension = extension[1:]

    return os.path.splitext(file_path)[0] + '.' + extension


def file_exists_with_diff_extension(file_path, extension):
    return file_path is None or \
           os.path.isfile(replace_file_extension(file_path, extension))


def add_sequence_feature_column(df, col_name, seq_length):
    """
    Adds a new column to the dataframe computed from an existing column.
    Values in the new column are space-delimited strings composed of preceding
    values of the same column up to seq_length.
    For example values of the i-th row of the new column will be a
    space-delimited string of df[col_name][i-seq_length].
     :param df: input dataframe
    :param col_name: column name containing sequential data
    :param seq_length: length of an array of preceeding column values to use
    """
    if col_name not in df.columns.values:
        logger.error('{} column does not exist'.format(col_name))
        return

    new_col_name = col_name + '_feature'
    if new_col_name in df.columns.values:
        logger.warning(
            '{} column already exists, values will be overridden'.format(
                new_col_name
            )
        )

    new_data = [None] * seq_length
    old_data = np.array(df[col_name])

    for i in range(seq_length, len(df)):
        new_data.append(' '.join(
            str(j) for j in old_data[i - seq_length: i]
        ))

    df[new_col_name] = new_data
    df[new_col_name] = df[new_col_name].fillna(method='backfill')


def override_in_memory_flag(input_features, override_value):
    num_overrides = 0
    for feature in input_features:
        if PREPROCESSING in feature:
            if 'in_memory' in feature[PREPROCESSING]:
                feature[PREPROCESSING]['in_memory'] = override_value
                num_overrides += 1
    return num_overrides


def normalize_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return normalize_numpy(obj.tolist())
    elif isinstance(obj, list):
        return [normalize_numpy(v) for v in obj]
    else:
        return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


def generate_kfold_splits(data_df, num_folds, random_state):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_num = 0
    for train_indices, test_indices in kf.split(data_df):
        fold_num += 1
        yield train_indices, test_indices, fold_num


def get_path_size(
        start_path,
        regex_accept=None,
        regex_reject=None
):
    total_size = 0
    pattern_accept = re.compile(regex_accept) if regex_accept else None
    pattern_reject = re.compile(regex_reject) if regex_reject else None

    for dirpath, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if not os.path.islink(filepath):
                accepted = True
                if pattern_accept:
                    accepted = accepted and pattern_accept.match(filename)
                if pattern_reject:
                    accepted = accepted and not pattern_reject.match(filename)
                if accepted:
                    total_size += os.path.getsize(filepath)

    return total_size


def clear_data_cache():
    """Clears any cached data objects (e.g., embeddings)"""
    load_glove.cache_clear()


def figure_data_format_dataset(dataset):
    if isinstance(dataset, pd.DataFrame):
        return pd.DataFrame
    elif isinstance(dataset, dd.core.DataFrame):
        return dd.core.DataFrame
    elif isinstance(dataset, dict):
        return dict
    elif isinstance(dataset, str):
        dataset = dataset.lower()
        if dataset.endswith('.csv'):
            return 'csv'
        elif dataset.endswith('.tsv'):
            return 'tsv'
        elif dataset.endswith('.json'):
            return 'json'
        elif dataset.endswith('.jsonl'):
            return 'jsonl'
        elif (dataset.endswith('.xls') or dataset.endswith('.xlsx') or
              dataset.endswith('.xlsm') or dataset.endswith('.xlsb') or
              dataset.endswith('.odf') or dataset.endswith('.ods') or
              dataset.endswith('.odt')):
            return 'excel'
        elif dataset.endswith('.parquet'):
            return 'parquet'
        elif dataset.endswith('.pickle') or dataset.endswith('.p'):
            return 'pickle'
        elif dataset.endswith('.feather'):
            return 'feather'
        elif dataset.endswith('.fwf'):
            return 'fwf'
        elif dataset.endswith('.html'):
            return 'html'
        elif dataset.endswith('.orc'):
            return 'orc'
        elif dataset.endswith('.sas'):
            return 'sas'
        elif dataset.endswith('.spss'):
            return 'spss'
        elif dataset.endswith('.dta') or dataset.endswith('.stata'):
            return 'stata'
        elif dataset.endswith('.h5') or dataset.endswith('.hdf5'):
            return 'hdf5'
        else:
            raise ValueError(
                "Dataset path string {} "
                "does not contain a valid extension".format(dataset)
            )
    else:
        raise ValueError(
            "Cannot figure out the format of dataset {}".format(dataset)
        )


def figure_data_format(
        dataset=None, training_set=None, validation_set=None, test_set=None
):
    if dataset is not None:
        data_format = figure_data_format_dataset(dataset)
    elif training_set is not None:
        data_formats = [figure_data_format_dataset(training_set)]
        if validation_set is not None:
            data_formats.append(figure_data_format_dataset(validation_set))
        if test_set is not None:
            data_formats.append(figure_data_format_dataset(test_set))
        data_formats_set = set(data_formats)
        if len(data_formats_set) > 1:
            error_message = "Datasets have different formats. Training: "
            error_message += str(data_formats[0])
            if validation_set:
                error_message = ", Validation: "
                error_message += str(data_formats[1])
            if test_set:
                error_message = ", Test: "
                error_message += str(data_formats[-1])
            raise ValueError(error_message)
        else:
            data_format = next(iter(data_formats_set))
    else:
        raise ValueError(
            "At least one between dataset and training_set must be not None"
        )
    return data_format


def is_model_dir(path: str) -> bool:
    hyperparameters_fn = os.path.join(path, MODEL_HYPERPARAMETERS_FILE_NAME)
    ts_metadata_fn = os.path.join(path, TRAIN_SET_METADATA_FILE_NAME)
    is_model_dir = False
    if (os.path.isdir(path)
            and os.path.isfile(hyperparameters_fn)
            and os.path.isfile(ts_metadata_fn)):
        weights_files_count = 0
        for file_name in os.listdir(path):
            if file_name.startswith(MODEL_WEIGHTS_FILE_NAME):
                weights_files_count += 1
        if weights_files_count >= 2:
            is_model_dir = True
    return is_model_dir


external_data_reader_registry = {
    **{fmt: read_csv for fmt in CSV_FORMATS},
    **{fmt: read_tsv for fmt in TSV_FORMATS},
    **{fmt: read_json for fmt in JSON_FORMATS},
    **{fmt: read_jsonl for fmt in JSONL_FORMATS},
    **{fmt: read_excel for fmt in EXCEL_FORMATS},
    **{fmt: read_parquet for fmt in PARQUET_FORMATS},
    **{fmt: read_pickle for fmt in PICKLE_FORMATS},
    **{fmt: read_fwf for fmt in FWF_FORMATS},
    **{fmt: read_feather for fmt in FEATHER_FORMATS},
    **{fmt: read_html for fmt in HTML_FORMATS},
    **{fmt: read_orc for fmt in ORC_FORMATS},
    **{fmt: read_sas for fmt in SAS_FORMATS},
    **{fmt: read_spss for fmt in SPSS_FORMATS},
    **{fmt: read_stata for fmt in STATA_FORMATS}
}
