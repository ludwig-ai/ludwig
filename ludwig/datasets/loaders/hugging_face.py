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
import datasets
import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class HFText2TextGenerationLoader(DatasetLoader):
    def load_hf_to_dataframe(self, hf_id: str, hf_subset: str, split: str) -> pd.DataFrame:
        dataset = datasets.load_dataset(path=hf_id, name=hf_subset, split=split)
        return dataset.to_pandas()
