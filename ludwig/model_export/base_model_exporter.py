import torch
from ludwig.api import LudwigModel
from abc import ABC, abstractmethod

class BaseModelExporter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def export_classifier_onnx(model_path, export_path):
        pass


    @abstractmethod
    def quantize(path_fp32, path_int8):
        pass