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

"""All contrib classes must have the following methods:

- import_call: Run on import.

Contrib classes can have the following methods:

- train: Run on train.

- visualize: Run on visualize.

- visualize_figure: Run when a figure is shown.

- experiment: Run on experiment.

- predict: Run on predict.

If you don't want to handle the call, either provide an empty
method with `pass`, or just don't implement the method.
"""

from .aim import AimCallback

# Contributors, import your class here:
from .comet import CometCallback
from .mlflow import MlflowCallback
from .wandb import WandbCallback

contrib_registry = {
    # Contributors, add your class here:
    "comet": CometCallback,
    "wandb": WandbCallback,
    "mlflow": MlflowCallback,
    "aim": AimCallback,
}
