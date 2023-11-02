from abc import ABC, abstractmethod

import torch

from ludwig.api import LudwigModel


class LudwigTorchWrapper(torch.nn.Module):
    def __init__(self, model):
        super(LudwigTorchWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model({"image_path": x})


class BaseModelExporter(ABC):

    @abstractmethod
    def export(self, model_path, export_path, export_args_override):
        pass


    @abstractmethod
    def check_model_export(self, path):
        pass