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

import tempfile
import pandas as pd
from ludwig.datasets.mnist import Mnist
from ludwig.data.preprocessing import get_split
from ludwig.utils.data_utils import split_dataset_ttv

class MNistDataSet(Mnist):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)


"""Lets write a simple driver class with a main program to test the mnist workflow"""


class MnistDriver:

    def load_mnist_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = MNistDataSet(tmpdir)
            dataset.load()
            final_df = dataset.prepare_final_dataset()
            training_set, test_set, validation_set = split_dataset_ttv(
                final_df,
                get_split(final_df)
            )
            training_set = pd.DataFrame(training_set)
            test_set = pd.DataFrame(test_set)
        return training_set, test_set


# we retrieve the training and test dataset, user can add more flavor
# as to what they want to do with that dataset
def main():
    mnist_driver_handle = MnistDriver()
    main_training_set, main_test_set = mnist_driver_handle.load_mnist_dataset()


if __name__ == "__main__":
    main()



