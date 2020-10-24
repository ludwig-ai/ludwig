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
import gzip
import os
import requests


class GZipDownloadMixin:
    """Downloads a gzip file containing the mnist png dataset and extracts its contents."""

    config: dict
    raw_dataset_path: str
    raw_temp_path: str

    def download_raw_dataset(self):
        """
        Download the raw dataset and contents of the gzip file
        onto the _raw directory.
        """
        file_download_list = {os.join(self.download_url + self.config['training_image_root']):
                                  (os.join(self.raw_temp_path + self.config['training_image_root'] + ".gz"),
                                   os.join(self.download_url + self.config['training_image_root'])),
                              os.join(self.download_url + self.config['training_label_root']):
                                  (os.join(self.raw_temp_path + self.config['training_label_root'] + ".gz"),
                                   os.join(self.download_url + self.config['training_label_root'])),
                              os.join(self.download_url + self.config['test_image_root']):
                                  (os.join(self.raw_temp_path + self.config['test_image_root'] + ".gz"),
                                   os.join(self.download_url + self.config['test_image_root'])),
                              os.join(self.download_url + self.config['test_label_root']):
                                  (os.join(self.raw_temp_path + self.config['test_label_root'] + ".gz"),
                                   os.join(self.download_url + self.config['test_label_root']))
                              }
        for file_download_url, download_file_tuple in file_download_list.items():
            download_archive = download_file_tuple[0]
            download_archive_contents = download_file_tuple[1]
            response = requests.get(file_download_url, stream=True)
            if response.status_code == 200:
                with open(download_archive, 'wb') as f:
                    f.write(response.raw.read())
            input_gzip_file = gzip.GzipFile(download_archive, 'rb')
            s = input_gzip_file.read()
            input_gzip_file.close()

            output = open(download_archive_contents, 'wb')
            output.write(s)

        os.rename(self.raw_temp_path, self.raw_dataset_path)

    @property
    def download_url(self):
        return self.config["download_url"]
