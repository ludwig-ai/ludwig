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
import os
import png
import struct
from os import path
from array import array
import pandas as pd
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.download import GZipDownloadMixin


def load(cache_dir=DEFAULT_CACHE_LOCATION):
    dataset = Mnist(cache_dir=cache_dir)
    return dataset.load()


class Mnist(CSVLoadMixin, GZipDownloadMixin, BaseDataset):
    """The Mnist dataset.

    This pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming training data into a destination
    dataframe that can fit into Ludwig's training API.
    """

    config: dict
    raw_temp_path: str
    raw_dataset_path: str
    processed_dataset_path: str

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="mnist", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        """Read the training and test directories and write out
        a csv containing the training path and the label.
        """
        for dataset in ["training", "testing"]:
            labels, data, rows, cols = self.read_source_dataset(dataset, self.raw_dataset_path)
            self.write_output_dataset(labels, data, rows, cols, path.join(self.raw_dataset_path, dataset))
        self.output_training_and_test_data(len(labels))
        os.rename(self.raw_dataset_path, self.processed_dataset_path)

    def prepare_final_dataset(self):
        """Given a training and test csv we want to create a single final
        dataframe that contains a split column with different values of each
        of these (0 for training) and (2 for test) and then return a single
        merged dataframe containing that column
        Returns:
            A final merged dataframe containing the split column"""
        training_df = pd.read_csv(os.path.join(self.processed_dataset_path, "mnist_dataset_training.csv"))
        training_df["split"] = 0
        test_df = pd.read_csv(os.path.join(self.processed_dataset_path, "mnist_dataset_testing.csv"))
        test_df["split"] = 2
        frames = [training_df, test_df]
        return pd.concat(frames)

    def read_source_dataset(self, dataset="training", path="."):
        """Create a directory for training and test and extract all the images
        and labels to this destination.
        :args:
            dataset (str) : the label for the dataset
            path (str): the raw dataset path
        :returns:
            A tuple of the label for the image, the file array, the size and rows and columns for the image"""
        if dataset is "training":
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif dataset is "testing":
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")
        flbl = open(fname_lbl, 'rb')
        struct.unpack(">II", flbl.read(8))
        lbl = array("b", flbl.read())
        flbl.close()

        fimg = open(fname_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = array("B", fimg.read())
        fimg.close()
        return lbl, img, rows, cols

    def write_output_dataset(self, labels, data, rows, cols, output_dir):
        """Create output directories where we write out the images.
        :args:
            labels (str) : the labels for the image
            data (np.array) : the binary array corresponding to the image
            rows (int) : the number of rows in the image
            cols (int) : the number of columns in the image
            output_dir (str) : the output directory that we need to write to
            path (str): the raw dataset path
        :returns:
            A tuple of the label for the image, the file array, the size and rows and columns for the image"""
        # create child image output directories
        output_dirs = [
            path.join(output_dir, str(i))
            for i in range(len(labels))
        ]
        for output_dir in output_dirs:
            if not path.exists(output_dir):
                os.makedirs(output_dir)

        # write out image data
        for (i, label) in enumerate(labels):
            output_filename = path.join(output_dirs[label], str(i) + ".png")
            with open(output_filename, "wb") as h:
                w = png.Writer(cols, rows, greyscale=True)
                data_i = [
                    data[(i * rows * cols + j * cols): (i * rows * cols + (j + 1) * cols)]
                    for j in range(rows)
                ]
                w.write(h, data_i)

    def output_training_and_test_data(self, length_labels):
        """The final method where we create a training and test file by iterating through
        all the images and labels previously created.
        Args:
            length_labels (int): The number of labels for the images that we're processing"""
        subdirectory_list = ["training",
                             "testing"]
        for name in subdirectory_list:
            with open(os.path.join(self.raw_dataset_path, 'mnist_dataset_{}.csv'.format(name)), 'w') as output_file:
                print('=== creating {} dataset ==='.format(name))
                output_file.write('image_path,label\n')
                for i in range(length_labels):
                    img_path = os.path.join(self.raw_dataset_path, '{}/{}'.format(name, i))
                    for file in os.listdir(img_path):
                        if file.endswith(".png"):
                            output_file.write('{},{}\n'.format(os.path.join(img_path, file), str(i)))

    @property
    def download_url(self):
        return self.config["download_url"]
