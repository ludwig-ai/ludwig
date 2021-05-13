#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2020 Uber Technologies, Inc.
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

from ludwig.backend.base import Backend, LocalTrainingMixin
from ludwig.constants import NAME, PARQUET
from ludwig.data.dataframe.dask import DaskEngine
from ludwig.data.dataset.partitioned import PartitionedDataset
from ludwig.models.predictor import BasePredictor, Predictor, get_output_columns


class DaskRemoteModel:
    def __init__(self, model):
        self.cls, self.args, state = list(model.__reduce__())
        self.state = state

    def load(self):
        obj = self.cls(*self.args)
        # TODO(travis): get_connected_model is needed here because TF will not init
        #  all weights until the graph has been traversed
        obj.get_connected_model()
        obj.__setstate__(self.state)
        return obj


class DaskPredictor(BasePredictor):
    def __init__(self, predictor_kwargs):
        self.predictor_kwargs = predictor_kwargs

    def batch_predict(self, model, dataset, *args, **kwargs):
        self._check_dataset(dataset)

        remote_model = DaskRemoteModel(model)
        predictor_kwargs = self.predictor_kwargs
        output_columns = get_output_columns(model.output_features)

        def batch_predict_partition(dataset):
            model = remote_model.load()
            predictor = Predictor(**predictor_kwargs)
            predictions = predictor.batch_predict(model, dataset, *args, **kwargs)
            ordered_predictions = predictions[output_columns]
            return ordered_predictions

        return dataset.map_dataset_partitions(
            batch_predict_partition,
            meta=[(c, 'object') for c in output_columns]
        )

    def batch_evaluation(self, model, dataset, collect_predictions=False, **kwargs):
        raise NotImplementedError(
            'Dask backend does not support batch evaluation at this time.'
        )

    def batch_collect_activations(self, model, *args, **kwargs):
        raise NotImplementedError(
            'Dask backend does not support collecting activations at this time.'
        )

    def _check_dataset(self, dataset):
        if not isinstance(dataset, PartitionedDataset):
            raise RuntimeError(
                f'Dask backend requires PartitionedDataset for inference, '
                f'found: {type(dataset)}'
            )

    def shutdown(self):
        pass


class DaskBackend(LocalTrainingMixin, Backend):
    def __init__(self, data_format=PARQUET, **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self._df_engine = DaskEngine()
        if data_format != PARQUET:
            raise ValueError(
                f'Data format {data_format} is not supported when using the Dask backend. '
                f'Try setting to `parquet`.'
            )

    def initialize(self):
        pass

    def create_predictor(self, **kwargs):
        return DaskPredictor(kwargs)

    @property
    def df_engine(self):
        return self._df_engine

    @property
    def supports_multiprocessing(self):
        return False

    def check_lazy_load_supported(self, feature):
        raise ValueError(f'DaskBackend does not support lazy loading of data files at train time. '
                         f'Set preprocessing config `in_memory: True` for feature {feature[NAME]}')
