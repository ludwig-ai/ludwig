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
import logging
import os
import re
from sklearn.decomposition import PCA
import struct
from multiprocessing.pool import ThreadPool
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from ludwig.utils.fs_utils import makedirs

logger = logging.getLogger(__name__)
NUM_LABELS = 10


class AV_MNISTLoader(DatasetLoader):
    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, transform = None, modal_separate = None, modal = None):
        try:
            from torchvision.io import write_png

            self.write_png = write_png
            self.transform = transform
            self.modal_separate = modal_separate
            self.modal = modal
            self.audio_data = None
            self.mnist_data = None
            self.labels = None
            self.data = None
        except ImportError:
            logger.error(
                "torchvision is not installed. "
                "In order to install all image feature dependencies run "
                "pip install ludwig[image]"
            )
            raise
        super().__init__(config, cache_dir)

    def transform_files(self,
                        root_dir,
                        file_paths: List[str],
                        modal_separate=True,
                        transform=None,
                        modal='image') -> List[str]:
        """
        Args:
            root_dir (str): The root directory where the processing is to be done
            file_paths (List[string]): The directory where all the .npy files are located for the training and test data.
            modal_separate (bool): define whether we're dealing with a modal or note
            transform
        Returns:
            The array of directory paths
        """
        self.process_source_dataset(self.raw_dataset_dir)
        for dataset in ["training", "testing"]:
            self.transform = transform
            self.modal_separate = modal_separate
            self.modal = modal
            if not modal_separate:
                if dataset == 'train':
                    self.audio_data = np.load(os.path.join(root_dir, 'audio', 'train_data.npy'))
                    self.mnist_data = np.load(os.path.join(root_dir, 'image', 'train_data.npy'))
                    self.labels = np.load(os.path.join(root_dir, 'train_labels.npy'))
                else:
                    self.audio_data = np.load(os.path.join(root_dir, 'audio', 'test_data.npy'))
                    self.mnist_data = np.load(os.path.join(root_dir, 'image', 'test_data.npy'))
                    self.labels = np.load(os.path.join(root_dir, 'test_labels.npy'))

                self.audio_data = self.audio_data[:, np.newaxis, :, :]
                self.mnist_data = self.mnist_data.reshape(self.mnist_data.shape[0], 1, 28, 28)
                self.write_output_dataset(self.labels, self.audio_data, os.path.join(self.raw_dataset_dir, dataset))
                self.write_output_dataset(self.labels, self.mnist_data, os.path.join(self.raw_dataset_dir, dataset))
            else:
                if modal:
                    if modal not in ['audio', 'image']:
                        raise ValueError('the value of modal is allowed')

                    if dataset == 'train':
                        self.data = np.load(os.path.join(root_dir, modal, 'train_data.npy'))
                        self.labels = np.load(os.path.join(root_dir, 'train_labels.npy'))
                    else:
                        self.data = np.load(os.path.join(root_dir, modal, 'test_data.npy'))
                        self.labels = np.load(os.path.join(root_dir, 'test_labels.npy'))

                    if modal == 'audio':
                        self.data = self.data[:, np.newaxis, :, :]
                    elif modal == 'image':
                        self.data = self.data.reshape(self.data.shape[0], 1, 28, 28)
                    self.write_output_dataset(self.labels, self.data, os.path.join(self.raw_dataset_dir, dataset))

                else:
                    raise ValueError('the value of modal should be given')

        return super().transform_files(file_paths)

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        """Load dataset files into a dataframe."""
        return self.output_training_and_test_data()

    def read_source_dataset(self, dataset="training", path="."):
        """Create a directory for training and test and extract all the images and labels to this destination.

        :args:
            dataset (str) : the label for the dataset
            path (str): the raw dataset path
        :returns:
            A tuple of the label for the image, the file array, the size and rows and columns for the image
        """
        if dataset == "training":
            fname_img = os.path.join(path, "train-images-idx3-ubyte")
            fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
        elif dataset == "testing":
            fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
            fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        with open(fname_lbl, "rb") as flbl:
            struct.unpack(">II", flbl.read(8))
            lbl = np.frombuffer(flbl.read(), dtype=np.uint8)

        with open(fname_img, "rb") as fimg:
            magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.frombuffer(fimg.read(), dtype=np.uint8)
            img = img.reshape((size, rows, cols))

        return lbl, img

    def process_source_dataset(self, raw_dataset_dir="."):
        """An intermediate method to write .npy files for the training and the test image and label data to the raw dataset directory
        :args:
            raw_dataset_dir (str) : the directory where all the raw data lives
        :returns:
            None"""
        file_names = {'train_data': 'train-images-idx3-ubyte.gz', 'train_labels': 'train-labels-idx1-ubyte.gz',
                      'test_data': 't10k-images-idx3-ubyte.gz', 'test_labels': 't10k-labels-idx1-ubyte.gz'}
        print("Raw dataset working directory: %s" % raw_dataset_dir)

        for key, file_name in file_names.items():
            file_path = os.path.join(raw_dataset_dir, file_name)
            print('file: %s' % key)
            f = open(file_path, "r")
            # read the definition of idx1-ubyte and idx3-ubyte
            f.seek(4)
            num = f.read(4)
            num = int().from_bytes(num, 'big')
            print('size of %s : %d' % (key, num))
            if re.match(r'.*data.*', key) is not None:
                height = f.read(4)
                height = int().from_bytes(height, 'big')
                width = f.read(4)
                width = int().from_bytes(width, 'big')

                data = np.frombuffer(f.read(), np.uint8).reshape(num, height, width)

                # PCA projecting with 75% energy removing
                n_comp = int(height * width)
                pca = PCA(n_components=n_comp)
                projected = pca.fit_transform(data.reshape(num, height * width))
                n_comp = ((np.cumsum(pca.explained_variance_ratio_) > 0.25) != 0).argmax()
                rec = np.matmul(projected[:, :n_comp], pca.components_[:n_comp])

                saved_path = os.path.join(raw_dataset_dir, "images")
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                saved_name = key + '.npy'
                np.save(os.path.join(saved_path, saved_name), rec)
            else:
                data = np.frombuffer(f.read(), np.uint8)

                saved_path = os.path.join(raw_dataset_dir, "labels")
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                saved_name = key + '.npy'
                np.save(os.path.join(saved_path, saved_name), data)

    def read_source_dataset(self, dataset="training", path="."):
        """Create a directory for training and test and extract all the images and labels to this destination.

        :args:
            dataset (str) : the label for the dataset
            path (str): the raw dataset path
        :returns:
            A tuple of the label for the image, the file array, the size and rows and columns for the image
        """
        if dataset == "training":
            fname_img = os.path.join(path, "train-images-idx3-ubyte")
            fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
        elif dataset == "testing":
            fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
            fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        with open(fname_lbl, "rb") as flbl:
            struct.unpack(">II", flbl.read(8))
            lbl = np.frombuffer(flbl.read(), dtype=np.uint8)

        with open(fname_img, "rb") as fimg:
            magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.frombuffer(fimg.read(), dtype=np.uint8)
            img = img.reshape((size, rows, cols))

        return lbl, img

    def write_output_dataset(self, labels, images, output_dir):
        """Create output directories where we write out the images.

        :args:
            labels (str) : the labels for the image
            data (np.array) : the binary array corresponding to the image
            output_dir (str) : the output directory that we need to write to
            path (str): the raw dataset path
        :returns:
            A tuple of the label for the image, the file array, the size and rows and columns for the image
        """
        # create child image output directories
        output_dirs = [os.path.join(output_dir, str(i)) for i in range(NUM_LABELS)]

        for output_dir in output_dirs:
            makedirs(output_dir, exist_ok=True)

        def write_processed_image(t):
            i, label = t
            output_filename = os.path.join(output_dirs[label], str(i) + ".png")
            torch_image = torch.from_numpy(images[i].copy()).view(1, 28, 28)
            self.write_png(torch_image, output_filename)

        # write out image data
        tasks = list(enumerate(labels))
        pool = ThreadPool(NUM_LABELS)
        pool.map(write_processed_image, tasks)
        pool.close()
        pool.join()

    def output_training_and_test_data(self):
        """Creates a combined (training and test) dataframe by iterating through all the images and labels."""
        dataframes = []
        for name in ["training", "testing"]:
            labels = []
            paths = []
            splits = []
            for i in range(NUM_LABELS):
                label_dir = f"{name}/{i}"
                img_dir = os.path.join(self.processed_dataset_dir, label_dir)
                for file in os.listdir(img_dir):
                    if file.endswith(".png"):
                        labels.append(str(i))
                        paths.append(os.path.join(label_dir, file))
                        splits.append(0 if name == "training" else 2)
            dataframes.append(pd.DataFrame({"image_path": paths, "label": labels, "split": splits}))
        return pd.concat(dataframes, ignore_index=True)
