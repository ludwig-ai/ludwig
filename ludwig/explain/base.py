from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.explain.util import Explanation, prepare_data
from ludwig.utils.torch_utils import get_torch_device

DEVICE = get_torch_device()


class Explainer(metaclass=ABCMeta):
    def __init__(self, model: LudwigModel, inputs_df: pd.DataFrame, sample_df: pd.DataFrame, target: str):
        """Constructor for the explainer.

        # Inputs

        :param model: (LudwigModel) The LudwigModel to explain.
        :param inputs_df: (pd.DataFrame) The input data to explain.
        :param sample_df: (pd.DataFrame) A sample of the ground truth data.
        :param target: (str) The name of the target to explain.
        """
        model.model.to(DEVICE)

        self.model = model
        self.inputs_df = inputs_df
        self.sample_df = sample_df
        self.target = target
        self.inputs_df, self.sample_df, self.feature_cols, self.target_feature_name = prepare_data(
            model, inputs_df, sample_df, target
        )

        self.explanations = [Explanation(self.target_feature_name) for _ in self.inputs_df.index]

    @abstractmethod
    def explain(self, **kwargs) -> Tuple[List[Explanation], List[float]]:
        """Explain the model's predictions.

        # Return

        :return: (Tuple[List[Explanation], List[float]]) `(explanations, expected_values)`
            `explanations`: (List[Explanation]) A list of explanations, one for each row in the input data. Each
            explanation contains the feature attributions for each label in the target feature's vocab.

            `expected_values`: (List[float]) of length [output feature cardinality] Expected value for each label in
            the target feature's vocab.
        """
