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
from ludwig.constants import BAG
from ludwig.constants import BINARY
from ludwig.constants import CATEGORY
from ludwig.constants import IMAGE
from ludwig.constants import NUMERICAL
from ludwig.constants import SEQUENCE
from ludwig.constants import SET
from ludwig.constants import TEXT
from ludwig.constants import TIMESERIES
from ludwig.features.bag_feature import BagBaseFeature
from ludwig.features.bag_feature import BagInputFeature
from ludwig.features.binary_feature import BinaryBaseFeature
from ludwig.features.binary_feature import BinaryInputFeature
from ludwig.features.binary_feature import BinaryOutputFeature
from ludwig.features.category_feature import CategoryBaseFeature
from ludwig.features.category_feature import CategoryInputFeature
from ludwig.features.category_feature import CategoryOutputFeature
from ludwig.features.image_feature import ImageBaseFeature
from ludwig.features.image_feature import ImageInputFeature
from ludwig.features.numerical_feature import NumericalBaseFeature
from ludwig.features.numerical_feature import NumericalInputFeature
from ludwig.features.numerical_feature import NumericalOutputFeature
from ludwig.features.sequence_feature import SequenceBaseFeature
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.sequence_feature import SequenceOutputFeature
from ludwig.features.set_feature import SetBaseFeature
from ludwig.features.set_feature import SetInputFeature
from ludwig.features.set_feature import SetOutputFeature
from ludwig.features.text_feature import TextBaseFeature
from ludwig.features.text_feature import TextInputFeature
from ludwig.features.text_feature import TextOutputFeature
from ludwig.features.timeseries_feature import TimeseriesBaseFeature
from ludwig.features.timeseries_feature import TimeseriesInputFeature

base_type_registry = {
    TEXT: TextBaseFeature,
    CATEGORY: CategoryBaseFeature,
    SET: SetBaseFeature,
    BAG: BagBaseFeature,
    BINARY: BinaryBaseFeature,
    NUMERICAL: NumericalBaseFeature,
    SEQUENCE: SequenceBaseFeature,
    TIMESERIES: TimeseriesBaseFeature,
    IMAGE: ImageBaseFeature
}
input_type_registry = {
    TEXT: TextInputFeature,
    NUMERICAL: NumericalInputFeature,
    BINARY: BinaryInputFeature,
    CATEGORY: CategoryInputFeature,
    SET: SetInputFeature,
    SEQUENCE: SequenceInputFeature,
    IMAGE: ImageInputFeature,
    TIMESERIES: TimeseriesInputFeature,
    BAG: BagInputFeature
}
output_type_registry = {
    CATEGORY: CategoryOutputFeature,
    BINARY: BinaryOutputFeature,
    NUMERICAL: NumericalOutputFeature,
    SEQUENCE: SequenceOutputFeature,
    SET: SetOutputFeature,
    TEXT: TextOutputFeature
}
