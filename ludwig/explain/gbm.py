from typing import List, Tuple

import numpy as np

from ludwig.explain.base import Explainer
from ludwig.models.gbm import GBM


class GBMExplainer(Explainer):
    def explain(self, **kwargs) -> Tuple[np.array, List[float]]:
        """Explain the model's predictions using Integrated Gradients.

        Returns:
            A tuple of (attribution, expected values):
            attribution: (np.array) of shape [batch size, output feature cardinality, num input features]
                Attribution value for each possible output feature label with respect to each input feature for each row
                in inputs_df.
            expected values: (List[float]) of length [output feature cardinality]
                Expected value for each possible output feature label.
        """
        base_model: GBM = self.model.model
        bst = base_model.lgb_booster
        if bst is None:
            raise ValueError("Model has not been trained yet.")

        # Get global feature importance from the model, use it for each row in the batch.
        feature_importance = bst.feature_importance(importance_type="split")
        feature_importance = feature_importance / feature_importance.sum()

        attribution = None
        expected_values = None

        return attribution, expected_values
