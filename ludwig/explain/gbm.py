from typing import List, Tuple

from ludwig.explain.base import Explainer
from ludwig.explain.util import Explanation
from ludwig.models.gbm import GBM


class GBMExplainer(Explainer):
    def explain(self, **kwargs) -> Tuple[List[Explanation], List[float]]:
        """Explain the model's predictions. Uses the feature importances from the model.

        # Return

        :return: (Tuple[List[Explanation], List[float]]) `(explanations, expected_values)`
            `explanations`: (List[Explanation]) A list of explanations, one for each row in the input data. Each
            explanation contains the feature attributions for each label in the target feature's vocab.

            `expected_values`: (List[float]) of length [output feature cardinality] Expected value for each label in
            the target feature's vocab.
        """
        base_model: GBM = self.model.model
        bst = base_model.lgb_booster
        if bst is None:
            raise ValueError("Model has not been trained yet.")

        # Get global feature importance from the model, use it for each row in the batch.
        feat_imp = bst.feature_importance(importance_type="gain")
        # Scale the feature importance to sum to 1.
        feat_imp = feat_imp / feat_imp.sum() if feat_imp.sum() > 0 else feat_imp

        expected_values = []
        for target_idx in range(self.vocab_size):
            for explanation in self.explanations:
                # Add the feature attributions to the explanation object for this row.
                explanation.add(feat_imp)

            # TODO:
            expected_values.append(0.0)

        return self.explanations, expected_values
