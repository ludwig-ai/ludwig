import numpy as np

from ludwig.api_annotations import PublicAPI
from ludwig.explain.explainer import Explainer
from ludwig.explain.explanation import ExplanationsResult
from ludwig.models.gbm import GBM


@PublicAPI(stability="experimental")
class GBMExplainer(Explainer):
    def explain(self) -> ExplanationsResult:
        """Explain the model's predictions. Uses the feature importances from the model.

        # Return

        :return: ExplanationsResult containing the explanations.
            `global_explanations`: (Explanation) Aggregate explanation for the entire input data.

            `row_explanations`: (List[Explanation]) A list of explanations, one for each row in the input data. Each
            explanation contains the feature attributions for each label in the target feature's vocab.

            `expected_values`: (List[float]) of length [output feature cardinality] Expected value for each label in
            the target feature's vocab.
        """
        base_model: GBM = self.model.model
        gbm = base_model.lgbm_model
        if gbm is None:
            raise ValueError("Model has not been trained yet.")

        # Get global feature importance from the model, use it for each row in the batch.
        # TODO(travis): support local feature importance
        raw_feat_imp = gbm.booster_.feature_importance(importance_type="gain")

        # For vector input features, the feature importance is given per element of the vector.
        # As such, to obtain the total importance for the feature, we need to sum over all the importance
        # values for every element of the vector.
        feat_imp = np.empty(len(base_model.input_features))
        raw_idx = 0
        for i, input_feature in enumerate(base_model.input_features.values()):
            # Length of the feature vector is the output shape of the encoder
            feature_length = input_feature.output_shape[0]
            raw_idx_end = raw_idx + feature_length

            # Reduce the importance values for every element in the vector down to the sum and
            # insert it as the feature level importance value
            feat_imp[i] = raw_feat_imp[raw_idx:raw_idx_end].sum()
            raw_idx = raw_idx_end

        # Logical check that at the end we reduced every element of the raw feature importance
        # into the feature level importance
        assert raw_idx == len(raw_feat_imp)

        # Scale the feature importance to sum to 1.
        feat_imp = feat_imp / feat_imp.sum() if feat_imp.sum() > 0 else feat_imp

        expected_values = []
        for _ in range(self.vocab_size):
            self.global_explanation.add(base_model.input_features.keys(), feat_imp)

            for explanation in self.row_explanations:
                # Add the feature attributions to the explanation object for this row.
                explanation.add(base_model.input_features.keys(), feat_imp)

            # TODO:
            expected_values.append(0.0)

        return ExplanationsResult(self.global_explanation, self.row_explanations, expected_values)
