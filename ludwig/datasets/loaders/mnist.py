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


class MNISTLoader(DatasetLoader):
    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None):
        try:
            from torchvision.io import write_png

            self.write_png = write_png
        except ImportError:
            logger.error(
                "torchvision is not installed. "
                "In order to install all image feature dependencies run "
                "pip install ludwig[image]"
            )
            raise
        super().__init__(config, cache_dir)

    def transform_files(self, file_paths: List[str]) -> List[str]:
        for dataset in ["training", "testing"]:
            labels, images = self.read_source_dataset(dataset, self.raw_dataset_dir)
            self.write_output_dataset(labels, images, os.path.join(self.raw_dataset_dir, dataset))
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
                        paths.append(os.path.join(img_dir, file))
                        splits.append(0 if name == "training" else 2)
            dataframes.append(pd.DataFrame({"image_path": paths, "label": labels, "split": splits}))
        return pd.concat(dataframes, ignore_index=True)
