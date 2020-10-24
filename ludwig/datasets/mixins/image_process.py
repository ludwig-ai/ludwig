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
import struct
from array import array
from os import path
import png
from ludwig.datasets.mixins.process import IdentityProcessMixin


class ImageProcessMixin(IdentityProcessMixin):
    """A mixin that downloads the mnist dataset and extracts a training and test csv file set"""

    raw_dataset_path: str
    processed_dataset_path: str

    """Read the training and test directories and write out 
    a csv containing the training path and the label.
    """
    def process_downloaded_dataset(self):
        for dataset in ["training", "testing"]:
            labels, data, rows, cols = self.read_source_dataset(dataset, self.raw_dataset_path)
            self.write_output_dataset(labels, data, rows, cols, path.join(self.raw_dataset_path, dataset))
        self.output_training_and_test_data()
        os.rename(self.raw_dataset_path, self.processed_dataset_path)

    """Create a directory for training and test and extract all the images
    and labels to this destination.
    :args:
        dataset (str) : the label for the dataset
        path (str): the raw dataset path
    :returns:
        A tuple of the label for the image, the file array, the size and rows and columns for the image"""
    def read_source_dataset(self, dataset="training", path="."):
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
    def write_output_dataset(self, labels, data, rows, cols, output_dir):
        # create child image output directories
        output_dirs = [
            path.join(output_dir, str(i))
            for i in range(10)
        ]
        for dir in output_dirs:
            if not path.exists(dir):
                os.makedirs(dir)

        # write out image data
        for (i, label) in enumerate(labels):
            output_filename = path.join(output_dirs[label], str(i) + ".png")
            # print("writing " + output_filename)
            with open(output_filename, "wb") as h:
                w = png.Writer(cols, rows, greyscale=True)
                data_i = [
                    data[(i * rows * cols + j * cols): (i * rows * cols + (j + 1) * cols)]
                    for j in range(rows)
                ]
                w.write(h, data_i)

    """The final method where we create a training and test file by iterating through
    all the images and labels previously created."""
    def output_training_and_test_data(self):
        subdirectory_list = ["training",
                             "testing"]
        for name in subdirectory_list:
            with open(self.raw_dataset_path + 'mnist_dataset_{}.csv'.format(name), 'w') as output_file:
                print('=== creating {} dataset ==='.format(name))
                output_file.write('image_path,label\n')
                for i in range(10):
                    csv_path = self.raw_dataset_path + '{}/{}'.format(name, i)
                    for file in os.listdir(csv_path):
                        if file.endswith(".png"):
                            output_file.write('{},{}\n'.format(os.path.join(path, file), str(i)))
