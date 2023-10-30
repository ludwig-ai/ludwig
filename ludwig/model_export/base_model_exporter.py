from abc import ABC, abstractmethod

import torch

from ludwig.api import LudwigModel


class LudwigTorchWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model({"image_path": x})


class BaseModelExporter(ABC):
    @abstractmethod
    def export_classifier(self, model_path, export_path, export_args_override):
        pass

    @abstractmethod
    def quantize(self, path_fp32, path_int8):
        pass

    @abstractmethod
    def check_model_export(self):
        pass
