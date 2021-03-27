#! /usr/bin/env python
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
import logging
import os

import numpy as np
import tensorflow as tf

from ludwig.constants import *
from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)


class ZScoreTransformer:
    def __init__(self, mean: float = None, std: float = None, **kwargs: dict):
        self.mu = mean
        self.sigma = std

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.sigma + self.mu

    @staticmethod
    def fit_transform_params(
            column: np.ndarray,
            backend: 'Backend'
    ) -> dict:
        compute = backend.df_engine.compute
        return {
            'mean': compute(column.astype(np.float32).mean()),
            'std': compute(column.astype(np.float32).std())
        }


class MinMaxTransformer:
    def __init__(self, min: float = None, max: float = None, **kwargs: dict):
        self.min_value = min
        self.max_value = max
        self.range = None if min is None or max is None else max - min

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_value) / self.range

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.range is None:
            raise ValueError(
                'Numeric transformer needs to be instantiated with '
                'min and max values.'
            )
        return x * self.range + self.min_value

    @staticmethod
    def fit_transform_params(
            column: np.ndarray,
            backend: 'Backend'
    ) -> dict:
        compute = backend.df_engine.compute
        return {
            'min': compute(column.astype(np.float32).min()),
            'max': compute(column.astype(np.float32).max())
        }


class Log1pTransformer:
    def __init__(self, **kwargs: dict):
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        if np.any(x <= 0):
            raise ValueError(
                'One or more values are non-positive.  '
                'log1p normalization is defined only for positive values.'
            )
        return np.log1p(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(x)

    @staticmethod
    def fit_transform_params(
            column: np.ndarray,
            backend: 'Backend'
    ) -> dict:
        return {}


class IdentityTransformer:
    def __init__(self, **kwargs):
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def fit_transform_params(
            column: np.ndarray,
            backend: 'Backend'
    ) -> dict:
        return {}


numeric_transformation_registry = {
    'minmax': MinMaxTransformer,
    'zscore': ZScoreTransformer,
    'log1p': Log1pTransformer,
    None: IdentityTransformer
}
