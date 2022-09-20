from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.explain.util import prepare_data
from ludwig.utils.torch_utils import get_torch_device

DEVICE = get_torch_device()


class Explainer(metaclass=ABCMeta):
    def __init__(self, model: LudwigModel, inputs_df: pd.DataFrame, sample_df: pd.DataFrame, target: str):
        """Initialize the explainer.

        Args:
            model: The LudwigModel to explain.
            inputs_df: The input data to explain.
            sample_df: A sample of the ground truth data.
            target: The name of the target to explain.
        """
        model.model.to(DEVICE)

        self.model = model
        self.inputs_df = inputs_df
        self.sample_df = sample_df
        self.target = target
        self.inputs_df, self.sample_df, self.feature_cols, self.target_feature_name = prepare_data(
            model, inputs_df, sample_df, target
        )

    @abstractmethod
    def explain(self, **kwargs) -> Tuple[np.array, List[float], np.array]:
        """Explain the model's predictions.

        Returns:
            A tuple of (attribution, expected values):
            attribution: (np.array) of shape [batch size, output feature cardinality, num input features]
                Attribution value for each possible output feature label with respect to each input feature for each
                row in inputs_df.
            expected values: (List[float]) of length [output feature cardinality]
                Expected value for each possible output feature label.
        """
