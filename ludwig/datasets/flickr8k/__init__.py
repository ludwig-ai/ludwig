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
from collections import defaultdict
import re

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import ZipDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin

def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = Flickr8k(cache_dir=cache_dir)
    return dataset.load(split=split)

class Flickr8k(CSVLoadMixin, ZipDownloadMixin, BaseDataset):
    """The Flickr8k dataset.

    This pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming training data into a destination
    dataframe that can fit into Ludwig's training API.
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="flickr8k", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        os.makedirs(self.processed_temp_path, exist_ok=True)
        # create a dictionary matching image_path --> list of captions
        image_to_caption = defaultdict(list)
        with open(
            f"{self.raw_dataset_path}/Flickr8k.token.txt",
            "r"
        ) as captions_file:
            image_to_caption = defaultdict(list)
            for line in captions_file:
                line = line.split("#")
                # the regex is to format the string to fit properly in a csv
                line[1] = line[1].strip("\n01234.\t ")
                line[1] = re.sub('\"', '\"\"', line[1])
                line[1] = '\"' + line[1] + '\"'
                image_to_caption[line[0]].append(line[1])
        # create csv file with 7 columns: image_path, 5 captions, and split
        with open(
                os.path.join(self.processed_temp_path, self.csv_filename),
                'w'
        ) as output_file:
            output_file.write('image_path,caption0,caption1,caption2,')
            output_file.write('caption3,caption4,split\n')
            splits = ["train", "dev", "test"]
            for i in range(len(splits)):
                split = splits[i]
                with open(
                    f"{self.raw_dataset_path}/Flickr_8k.{split}Images.txt",
                    "r"
                ) as split_file:
                    for image_name in split_file:
                        image_name = image_name.strip('\n')
                        if image_name in image_to_caption:
                            output_file.write('{},{},{},{},{},{},{}\n'.format(
                                # Note: image folder is named Flicker8k_Dataset
                                "{}/Flicker8k_Dataset/{}".format(
                                    self.raw_dataset_path, image_name
                                ),
                                *image_to_caption[image_name],
                                i
                            ))
        # Note: csv is stored in /processed while images are stored in /raw
        os.rename(self.processed_temp_path, self.processed_dataset_path)
