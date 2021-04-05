# !/usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Uber Technologies, Inc.
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

from abc import ABC


class Callback(ABC):
    def on_batch_start(self, trainer, progress_tracker, save_path):
        pass

    def on_batch_end(self, trainer, progress_tracker, save_path):
        pass

    def on_epoch_start(self, trainer, progress_tracker, save_path):
        pass

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        pass

    def on_validation_start(self, trainer, progress_tracker, save_path):
        pass

    def on_validation_end(self, trainer, progress_tracker, save_path):
        pass

    def on_test_start(self, trainer, progress_tracker, save_path):
        pass

    def on_test_end(self, trainer, progress_tracker, save_path):
        pass
