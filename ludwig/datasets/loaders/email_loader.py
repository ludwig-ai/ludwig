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
from typing import List, Optional

import pandas as pd
import tqdm

from ludwig.datasets.archives import is_archive
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from ludwig.datasets.loaders.email import email_features, email_parser

logger = logging.getLogger(__name__)


class EmailLoader(DatasetLoader):
    """Converts a folder of EML files to parquet dataset.

    If folders are present in the input directory, they are assumed to represent different classes and an additional
    column is added to the exported dataset using the directory names as labels.
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, add_binary_columns: bool = True):
        super().__init__(config, cache_dir=cache_dir)
        self.add_binary_columns = add_binary_columns

    def load_emails_from_files(self, email_files, labels):
        """Load a list of email files into a dataframe."""
        rows = []
        for filename, label in tqdm.tqdm(zip(email_files, labels), total=len(email_files)):
            try:
                message = email_parser.read_email(filename)
            except Exception as e:
                logger.warning(f"Failed to parse file, skipping: {filename}", str(e))
                continue
            # Extracts basic columns from email message (from, to, subject...).  Does not raise exceptions.
            message_columns = email_parser.message_to_columns(message, label)

            # Adds engineered features to list of columns.
            message_columns.update(email_features.features_from_message(message_columns, message))
            rows.append(message_columns)
        return pd.DataFrame(rows)

    def transform_files(self, file_paths: List[str]) -> List[str]:
        """Transform data files before loading to dataframe."""
        super().transform_files(file_paths)
        # Recursively walk subdirectories, build list of all directories in tree which also contain normal files.
        input_dir = os.path.normpath(self.raw_dataset_dir)
        files_to_load = []
        file_labels = []
        for root, dirs, files in os.walk(input_dir):
            for name in files:
                if not name.startswith(".") and not is_archive(name):
                    files_to_load.append(os.path.join(root, name))
                    # EML files are typically organized in a tree of directories.  Saves the relative path from dataset
                    # root to use as the label column.
                    file_labels.append(os.path.relpath(root, start=input_dir))

        df = self.load_emails_from_files(files_to_load, file_labels)
        data_file_path = os.path.join(self.raw_dataset_dir, self.processed_dataset_filename)
        df.to_parquet(data_file_path)
        return [data_file_path]

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        # Email datasets have already been packaged into a single file by transform_files.
        return self.load_file_to_dataframe(file_paths[0])

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = super().transform_dataframe(dataframe)
        if self.add_binary_columns:
            for label in df.label.unique():
                df[label] = df.label == label
        return df


class SpamAssassinLoader(EmailLoader):
    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None):
        super().__init__(config, cache_dir=cache_dir, add_binary_columns=True)

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Combine labels before adding binary columns.
        label_map = {
            "spam_2": "spam",
            "easy_ham_2": "easy_ham",
        }
        dataframe["label"] = dataframe.label.map(lambda l: label_map[l] if l in label_map else l)
        return super().transform_dataframe(dataframe)


class EnronLoader(EmailLoader):
    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None):
        super().__init__(config, cache_dir=cache_dir, add_binary_columns=False)

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = super().transform_dataframe(dataframe)
        del df["label"]  # Enron dataset has no labels
        return df
