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
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class ForestCoverLoader(DatasetLoader):
    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, use_tabnet_split=True):
        super().__init__(config, cache_dir=cache_dir)
        self.use_tabnet_split = use_tabnet_split

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = super().transform_dataframe(dataframe)
        # Elevation                               quantitative    meters                       Elevation in meters
        # Aspect                                  quantitative    azimuth                      Aspect in degrees azimuth
        # Slope                                   quantitative    degrees                      Slope in degrees
        # Horizontal_Distance_To_Hydrology        quantitative    meters                       Horz Dist to nearest surface water features      # noqa: E501
        # Vertical_Distance_To_Hydrology          quantitative    meters                       Vert Dist to nearest surface water features      # noqa: E501
        # Horizontal_Distance_To_Roadways         quantitative    meters                       Horz Dist to nearest roadway                     # noqa: E501
        # Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice          # noqa: E501
        # Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice          # noqa: E501
        # Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice          # noqa: E501
        # Horizontal_Distance_To_Fire_Points      quantitative    meters                       Horz Dist to nearest wildfire ignition points    # noqa: E501
        # Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation                      # noqa: E501
        # Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation
        # Cover_Type (7 types)                    integer         1 to 7                       Forest Cover Type designation                    # noqa: E501

        # Map the 40 soil types to a single integer instead of 40 binary columns
        st_cols = [
            "Soil_Type_1",
            "Soil_Type_2",
            "Soil_Type_3",
            "Soil_Type_4",
            "Soil_Type_5",
            "Soil_Type_6",
            "Soil_Type_7",
            "Soil_Type_8",
            "Soil_Type_9",
            "Soil_Type_10",
            "Soil_Type_11",
            "Soil_Type_12",
            "Soil_Type_13",
            "Soil_Type_14",
            "Soil_Type_15",
            "Soil_Type_16",
            "Soil_Type_17",
            "Soil_Type_18",
            "Soil_Type_19",
            "Soil_Type_20",
            "Soil_Type_21",
            "Soil_Type_22",
            "Soil_Type_23",
            "Soil_Type_24",
            "Soil_Type_25",
            "Soil_Type_26",
            "Soil_Type_27",
            "Soil_Type_28",
            "Soil_Type_29",
            "Soil_Type_30",
            "Soil_Type_31",
            "Soil_Type_32",
            "Soil_Type_33",
            "Soil_Type_34",
            "Soil_Type_35",
            "Soil_Type_36",
            "Soil_Type_37",
            "Soil_Type_38",
            "Soil_Type_39",
            "Soil_Type_40",
        ]
        st_vals = []
        for _, row in df[st_cols].iterrows():
            st_vals.append(row.to_numpy().nonzero()[0].item(0))
        df = df.drop(columns=st_cols)
        df["Soil_Type"] = st_vals

        # Map the 4 wilderness areas to a single integer
        # instead of 4 binary columns
        wa_cols = ["Wilderness_Area_1", "Wilderness_Area_2", "Wilderness_Area_3", "Wilderness_Area_4"]
        wa_vals = []
        for _, row in df[wa_cols].iterrows():
            wa_vals.append(row.to_numpy().nonzero()[0].item(0))
        df = df.drop(columns=wa_cols)
        df["Wilderness_Area"] = wa_vals

        if not self.use_tabnet_split:
            # first 11340 records used for training data subset
            # next 3780 records used for validation data subset
            # last 565892 records used for testing data subset
            df["split"] = [0] * 11340 + [1] * 3780 + [2] * 565892
        else:
            # Split used in the tabNet paper
            # https://github.com/google-research/google-research/blob/master/tabnet/download_prepare_covertype.py
            train_val_indices, test_indices = train_test_split(range(len(df)), test_size=0.2, random_state=0)
            train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2 / 0.6, random_state=0)

            df["split"] = 0
            df.loc[val_indices, "split"] = 1
            df.loc[test_indices, "split"] = 2

        return df
