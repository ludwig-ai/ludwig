# Copyright (c) 2023 Aizen Corp.
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
from typing import List

import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from ludwig.utils.fs_utils import makedirs


class CamseqLoader(DatasetLoader):
    def transform_files(self, file_paths: List[str]) -> List[str]:
        if not os.path.exists(self.processed_dataset_dir):
            os.makedirs(self.processed_dataset_dir)

        # move images and masks into separate directories
        source_dir = self.raw_dataset_dir
        images_dir = os.path.join(source_dir, "images")
        masks_dir = os.path.join(source_dir, "masks")
        makedirs(images_dir, exist_ok=True)
        makedirs(masks_dir, exist_ok=True)

        data_files = []
        for f in os.listdir(source_dir):
            if f.endswith("_L.png"):  # masks
                dest_file = os.path.join(masks_dir, f)
            elif f.endswith(".png"):  # images
                dest_file = os.path.join(images_dir, f)
            else:
                continue
            source_file = os.path.join(source_dir, f)
            os.replace(source_file, dest_file)
            data_files.append(dest_file)

        return super().transform_files(data_files)

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        """Creates a dataframe of image paths and mask paths."""
        images_dir = os.path.join(self.processed_dataset_dir, "images")
        masks_dir = os.path.join(self.processed_dataset_dir, "masks")
        images = []
        masks = []
        for f in os.listdir(images_dir):
            images.append(os.path.join(images_dir, f))
            mask_f = f[:-4] + "_L.png"
            masks.append(os.path.join(masks_dir, mask_f))

        return pd.DataFrame({"image_path": images, "mask_path": masks})
