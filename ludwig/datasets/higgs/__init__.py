#! /usr/bin/env python
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

import pandas as pd

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import GZipDownloadMixin
from ludwig.datasets.mixins.load import ParquetLoadMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs, rename


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, add_validation_set=False):
    dataset = Higgs(cache_dir=cache_dir, add_validation_set=add_validation_set)
    return dataset.load(split=split)


@register_dataset(name="higgs")
class Higgs(GZipDownloadMixin, ParquetLoadMixin, BaseDataset):
    """The Higgs Boson dataset.

    This is a classification problem to distinguish between a signal process
    which produces Higgs bosons and a background process which does not.

        More info:
        https://archive.ics.uci.edu/ml/datasets/HIGGS
    """

    raw_dataset_path: str
    processed_dataset_path: str

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, add_validation_set=False):
        super().__init__(dataset_name="higgs", cache_dir=cache_dir)
        self.add_validation_set = add_validation_set

    def process_downloaded_dataset(self):
        df = pd.read_csv(os.path.join(self.raw_dataset_path, "HIGGS.csv.gz"), header=None)

        df.columns = [
            "label",
            "lepton_pT",
            "lepton_eta",
            "lepton_phi",
            "missing_energy_magnitude",
            "missing_energy_phi",
            "jet_1_pt",
            "jet_1_eta",
            "jet_1_phi",
            "jet_1_b-tag",
            "jet_2_pt",
            "jet_2_eta",
            "jet_2_phi",
            "jet_2_b-tag",
            "jet_3_pt",
            "jet_3_eta",
            "jet_3_phi",
            "jet_3_b-tag",
            "jet_4_pt",
            "jet_4_eta",
            "jet_4_phi",
            "jet_4_b-tag",
            "m_jj",
            "m_jjj",
            "m_lv",
            "m_jlv",
            "m_bb",
            "m_wbb",
            "m_wwbb",
        ]

        df["label"] = df["label"].astype("int32")
        if self.add_validation_set:
            df["split"] = [0] * 10000000 + [1] * 500000 + [2] * 500000
        else:
            df["split"] = [0] * 10500000 + [2] * 500000

        makedirs(self.processed_temp_path, exist_ok=True)
        df.to_parquet(
            os.path.join(self.processed_temp_path, self.parquet_filename),
            engine="pyarrow",
            row_group_size=50000,
            index=False,
        )

        rename(self.processed_temp_path, self.processed_dataset_path)
