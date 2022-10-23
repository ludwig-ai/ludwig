from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from torch.autograd import Variable
from tqdm import tqdm

from ludwig.api import LudwigModel
from ludwig.api_annotations import PublicAPI
from ludwig.explain.captum import get_baseline, get_input_tensors, get_total_attribution, IntegratedGradientsExplainer
from ludwig.explain.explanation import Explanation
from ludwig.utils.torch_utils import DEVICE


@PublicAPI(stability="experimental")
class RayIntegratedGradientsExplainer(IntegratedGradientsExplainer):
    def __init__(self, resources_per_task: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resources_per_task = resources_per_task

    def explain(self) -> Tuple[List[Explanation], List[float]]:
        """Explain the model's predictions using Integrated Gradients.

        # Return

        :return: (Tuple[List[Explanation], List[float]]) `(explanations, expected_values)`
            `explanations`: (List[Explanation]) A list of explanations, one for each row in the input data. Each
            explanation contains the integrated gradients for each label in the target feature's vocab with respect to
            each input feature.

            `expected_values`: (List[float]) of length [output feature cardinality] Average convergence delta for each
            label in the target feature's vocab.
        """
        self.model.model.to(DEVICE)

        # Convert input data into embedding tensors from the output of the model encoders.
        inputs_encoded_ref = get_input_tensors_task.options(**self.resources_per_task).remote(
            ray.put(self.model), ray.put(self.inputs_df)
        )
        sample_encoded_ref = get_input_tensors_task.options(**self.resources_per_task).remote(
            ray.put(self.model), ray.put(self.sample_df)
        )

        inputs_encoded = ray.get(inputs_encoded_ref)
        sample_encoded = ray.get(sample_encoded_ref)
        baseline = get_baseline(sample_encoded)

        # Compute attribution for each possible output feature label separately.
        total_attribution_refs = []
        for target_idx in range(self.vocab_size):
            total_attribution_ref = get_total_attribution_task.options(**self.resources_per_task).remote(
                ray.put(self.model),
                self.target_feature_name,
                target_idx if self.is_category_target else None,
                ray.put(inputs_encoded),
                ray.put(baseline),
                self.use_global,
                len(self.inputs_df),
            )
            total_attribution_refs.append(total_attribution_ref)

            if self.is_binary_target:
                # For binary targets, we only need to compute attribution for the positive class (see below).
                break

        expected_values = []
        for total_attribution_ref in tqdm(total_attribution_refs, desc="Explain"):
            total_attribution = ray.get(total_attribution_ref)
            for feature_attributions, explanation in zip(total_attribution, self.explanations):
                # Add the feature attributions to the explanation object for this row.
                explanation.add(feature_attributions)

            # TODO(travis): for force plots, need something similar to SHAP E[X]
            expected_values.append(0.0)

        # For binary targets, add an extra attribution for the negative class (false).
        if self.is_binary_target:
            for explanation in self.explanations:
                le_true = explanation.label_explanations[0]
                explanation.add(le_true.feature_attributions * -1)

            # TODO(travis): for force plots, need something similar to SHAP E[X]
            expected_values.append(0.0)

        return self.explanations, expected_values


@ray.remote
def get_input_tensors_task(model: LudwigModel, df: pd.DataFrame) -> List[Variable]:
    model.model.to(DEVICE)
    return ray.put(get_input_tensors(model, df))


@ray.remote
def get_total_attribution_task(model: LudwigModel, *args, **kwargs) -> np.array:
    model.model.to(DEVICE)
    return ray.put(get_total_attribution(*args, **kwargs))
