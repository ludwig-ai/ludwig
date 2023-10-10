from abc import ABC, abstractmethod

import torch

from ludwig.api import LudwigModel


class BaseModelExporter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def export_classifier(self, model_path, export_path, export_args_override):
        pass

    @abstractmethod
    def quantize(self, path_fp32, path_int8):
        pass
