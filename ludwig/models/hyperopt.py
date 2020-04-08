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
from ludwig.utils.misc import get_from_registry


logger = logging.getLogger(__name__)


class RandomStrategy:
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass


class GridStrategy:
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass


class SerialExecutor:
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass


class ParallelExecutor:
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass


def get_build_hyperopt_strategy(strategy_type):
    return get_from_registry(
        strategy_type, strategy_registry
    )


def get_build_hyperopt_executor(executor_type):
    return get_from_registry(
        executor_type, executor_registry
    )


strategy_registry = {
    "random": RandomStrategy,
    "grid": GridStrategy
}

executor_registry = {
    "serial": SerialExecutor,
    "parallel": ParallelExecutor
}
