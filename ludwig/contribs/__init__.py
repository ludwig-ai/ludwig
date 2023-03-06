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

"""All contrib classes must implement the `ludwig.callbacks.Callback` interface.

If you don't want to handle the call, either provide an empty method with `pass`, or just don't implement the method.
"""

from abc import ABC, abstractmethod

from ludwig.callbacks import Callback


class ContribLoader(ABC):
    @abstractmethod
    def load(self) -> Callback:
        """Returns an instantiation of the callback instance, whose callback hooks will be invoked at runtime."""
        pass

    def preload(self):
        """Will always be called when Ludwig CLI is invoked, preload gives the callback an opportunity to import or
        create any shared resources.

        Importing required 3rd-party libraries should be done here i.e. import wandb. preload is guaranteed to be called
        before any other callback method, and will only be called once per process.
        """
        pass


# Contributors, load your class here:


class AimLoader(ContribLoader):
    def load(self) -> Callback:
        from ludwig.contribs.aim import AimCallback

        return AimCallback()

    def preload(self):
        import aim  # noqa


class CometLoader(ContribLoader):
    def load(self) -> Callback:
        from ludwig.contribs.comet import CometCallback

        return CometCallback()

    def preload(self):
        import comet_ml  # noqa


class WandbLoader(ContribLoader):
    def load(self) -> Callback:
        from ludwig.contribs.wandb import WandbCallback

        return WandbCallback()

    def preload(self):
        import wandb  # noqa


class MlflowLoader(ContribLoader):
    def load(self) -> Callback:
        from ludwig.contribs.mlflow import MlflowCallback

        return MlflowCallback()


contrib_registry = {
    # Contributors, add your class here:
    "comet": CometLoader(),
    "wandb": WandbLoader(),
    "mlflow": MlflowLoader(),
    "aim": AimLoader(),
}
