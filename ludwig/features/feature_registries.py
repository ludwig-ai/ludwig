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
from ludwig.constants import BAG, BINARY, CATEGORY, DATE, H3, IMAGE, NUMERICAL, \
    SEQUENCE, SET, TEXT, TIMESERIES, VECTOR, AUDIO
from ludwig.features.bag_feature import BagFeatureMixin, BagInputFeature
from ludwig.features.binary_feature import BinaryFeatureMixin, BinaryInputFeature, BinaryOutputFeature
from ludwig.features.category_feature import CategoryFeatureMixin, CategoryInputFeature, CategoryOutputFeature
from ludwig.features.date_feature import DateFeatureMixin, DateInputFeature
from ludwig.features.h3_feature import H3FeatureMixin, H3InputFeature
from ludwig.features.image_feature import ImageFeatureMixin, ImageInputFeature
from ludwig.features.numerical_feature import NumericalFeatureMixin, NumericalInputFeature, NumericalOutputFeature
from ludwig.features.audio_feature import AudioFeatureMixin, AudioInputFeature
from ludwig.features.sequence_feature import SequenceFeatureMixin, SequenceInputFeature, SequenceOutputFeature
from ludwig.features.set_feature import SetFeatureMixin, SetInputFeature, SetOutputFeature
from ludwig.features.text_feature import TextFeatureMixin, TextInputFeature, TextOutputFeature
from ludwig.features.timeseries_feature import TimeseriesFeatureMixin, TimeseriesInputFeature
from ludwig.features.vector_feature import VectorFeatureMixin, VectorInputFeature, VectorOutputFeature

base_type_registry = {
    TEXT: TextFeatureMixin,
    CATEGORY: CategoryFeatureMixin,
    SET: SetFeatureMixin,
    BAG: BagFeatureMixin,
    BINARY: BinaryFeatureMixin,
    NUMERICAL: NumericalFeatureMixin,
    SEQUENCE: SequenceFeatureMixin,
    TIMESERIES: TimeseriesFeatureMixin,
    IMAGE: ImageFeatureMixin,
    AUDIO: AudioFeatureMixin,
    H3: H3FeatureMixin,
    DATE: DateFeatureMixin,
    VECTOR: VectorFeatureMixin
}
input_type_registry = {
    TEXT: TextInputFeature,
    NUMERICAL: NumericalInputFeature,
    BINARY: BinaryInputFeature,
    CATEGORY: CategoryInputFeature,
    SET: SetInputFeature,
    SEQUENCE: SequenceInputFeature,
    IMAGE: ImageInputFeature,
    AUDIO: AudioInputFeature,
    TIMESERIES: TimeseriesInputFeature,
    BAG: BagInputFeature,
    H3: H3InputFeature,
    DATE: DateInputFeature,
    VECTOR: VectorInputFeature
}
output_type_registry = {
    CATEGORY: CategoryOutputFeature,
    BINARY: BinaryOutputFeature,
    NUMERICAL: NumericalOutputFeature,
    SEQUENCE: SequenceOutputFeature,
    SET: SetOutputFeature,
    TEXT: TextOutputFeature,
    VECTOR: VectorOutputFeature
}
