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

import pandas as pd
import ray
from functools import partial

from ludwig.backend.base import Backend, LocalTrainingMixin
from ludwig.constants import NAME, PARQUET, PREPROCESSING, TFRECORD
from ludwig.data.dataframe.dask import DaskEngine
from ludwig.data.dataset.pandas import PandasDataset
from ludwig.data.dataset.partitioned import RayDataset
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
        batch_predictor = self.BatchInferModel(
            remote_model, predictor_kwargs, output_columns, dataset.features,
            dataset.data_hdf5_fp, *args, **kwargs
        )

        num_gpus = int(ray.cluster_resources().get('GPU', 0) > 0)
        return dataset.ds.map_batches(
            batch_predictor,
            compute='actors',
            batch_format='pandas',
            num_gpus=num_gpus
        ).to_dask()

    def batch_evaluation(self, model, dataset, collect_predictions=False, **kwargs):
        raise NotImplementedError(
            'Dask backend does not support batch evaluation at this time.'
        )

    def batch_collect_activations(self, model, *args, **kwargs):
        raise NotImplementedError(
            'Dask backend does not support collecting activations at this time.'
        )

    def _check_dataset(self, dataset):
        if not isinstance(dataset, RayDataset):
            raise RuntimeError(
                f'Dask backend requires RayDataset for inference, '
                f'found: {type(dataset)}'
            )

    def shutdown(self):
        pass

    class BatchInferModel:
        def __init__(
                self, remote_model, predictor_kwargs, output_columns, features,
                data_hdf5_fp, *args, **kwargs
        ):
            self.model = remote_model.load()
            self.predictor = Predictor(**predictor_kwargs)
            self.output_columns = output_columns
            self.features = features
            self.data_hdf5_fp = data_hdf5_fp
            self.batch_predict = partial(self.predictor.batch_predict, *args,
                                         **kwargs)

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            pd_ds = PandasDataset(df, self.features, self.data_hdf5_fp)
            predictions = self.predictor.batch_predict(
                model=self.model, dataset=pd_ds)
            ordered_predictions = predictions[self.output_columns]
            return ordered_predictions


class DaskBackend(LocalTrainingMixin, Backend):
    def __init__(self, cache_format=PARQUET, engine=None, **kwargs):
        super().__init__(cache_format=cache_format, **kwargs)
        engine = engine or {}
        self._df_engine = DaskEngine(**engine)
        if cache_format not in [PARQUET, TFRECORD]:
            raise ValueError(
                f'Data format {cache_format} is not supported when using the Dask backend. '
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
        if not feature[PREPROCESSING]['in_memory']:
            raise ValueError(
                f'DaskBackend does not support lazy loading of data files at train time. '
                f'Set preprocessing config `in_memory: True` for feature {feature[NAME]}')
