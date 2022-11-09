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
import os

import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class AllstateClaimsSeverityLoader(DatasetLoader):
    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if os.path.basename(file_path) == "train.csv":
            # train.csv has been updated with quoted test rows at the end; don't load these, only load the original
            # training set.
            return pd.read_csv(file_path, nrows=188319)
        if os.path.basename(file_path) == "test.csv":
            # we limit the loaded rows for the same reason as the training set.
            return pd.read_csv(file_path, nrows=125547)
        super().load_file_to_dataframe(file_path)
