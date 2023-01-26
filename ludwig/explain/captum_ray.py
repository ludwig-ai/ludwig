from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from torch.autograd import Variable
from tqdm import tqdm

from ludwig.api import LudwigModel
from ludwig.api_annotations import PublicAPI
from ludwig.explain.captum import get_baseline, get_input_tensors, get_total_attribution, IntegratedGradientsExplainer
from ludwig.explain.explanation import Explanation
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.utils.torch_utils import get_torch_device


@PublicAPI(stability="experimental")
class RayIntegratedGradientsExplainer(IntegratedGradientsExplainer):
    def __init__(self, *args, resources_per_task: Dict[str, Any] = None, num_workers: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.resources_per_task = resources_per_task or {}
        self.num_workers = num_workers

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
        self.model.model.cpu()
        input_features: LudwigFeatureDict = self.model.model.input_features
        model_ref = ray.put(self.model)

        # Convert input data into embedding tensors from the output of the model encoders.
        inputs_encoded_ref = get_input_tensors_task.options(**self.resources_per_task).remote(
            model_ref, ray.put(self.inputs_df)
        )
        sample_encoded_ref = get_input_tensors_task.options(**self.resources_per_task).remote(
            model_ref, ray.put(self.sample_df)
        )

        inputs_encoded = ray.get(inputs_encoded_ref)
        sample_encoded = ray.get(sample_encoded_ref)
        baseline = get_baseline(self.model, sample_encoded)

        inputs_encoded_ref = ray.put(inputs_encoded)
        baseline_ref = ray.put(baseline)

        if self.is_category_target:
            # Evenly divide the list of labels among the desired number of workers (Ray tasks).
            # For example, 4 GPUs -> 4 workers. We do this instead of creating nlabels tasks because
            # there is significant overhead to spawning a Ray task.
            target_splits = split_list(list(range(self.vocab_size)), self.num_workers)
        else:
            # No target index to compare against exists for number features.
            # For binary targets, we only need to compute attribution for the positive class (see below).
            # May need to revisit in the future for additional feature types.
            target_splits = [[None]]

        # Compute attribution for each possible output feature label separately.
        attrs_refs = []
        for target_indices in target_splits:
            attrs_ref = get_total_attribution_task.options(**self.resources_per_task).remote(
                model_ref,
                self.target_feature_name,
                target_indices,
                inputs_encoded_ref,
                baseline_ref,
                self.use_global,
                len(self.inputs_df),
            )
            attrs_refs.append(attrs_ref)

        # Await the completion of our Ray tasks, then merge the results.
        expected_values = []
        for attrs_ref in tqdm(attrs_refs, desc="Explain"):
            attrs = ray.get(attrs_ref)
            for (total_attribution, feat_to_token_attribution) in attrs:
                for i, (feature_attributions, explanation) in enumerate(zip(total_attribution, self.explanations)):
                    # Add the feature attributions to the explanation object for this row.
                    explanation.add(
                        input_features.keys(),
                        feature_attributions,
                        {k: v[i] for k, v in feat_to_token_attribution.items()},
                    )

                # TODO(travis): for force plots, need something similar to SHAP E[X]
                expected_values.append(0.0)

        # For binary targets, add an extra attribution for the negative class (false).
        if self.is_binary_target:
            for explanation in self.explanations:
                le_true = explanation.label_explanations[0]
                negated_attributions = le_true.to_array() * -1
                negated_token_attributions = {
                    fa.feature_name: [(t, -a) for t, a in fa.token_attributions]
                    for fa in le_true.feature_attributions
                    if fa.token_attributions is not None
                }
                # Prepend the negative class to the list of label explanations.
                explanation.add(input_features.keys(), negated_attributions, negated_token_attributions, prepend=True)

            # TODO(travis): for force plots, need something similar to SHAP E[X]
            expected_values.append(0.0)

        return self.explanations, expected_values


@ray.remote(max_calls=1)
def get_input_tensors_task(model: LudwigModel, df: pd.DataFrame) -> List[Variable]:
    model.model.unskip()
    model.model.to(get_torch_device())
    try:
        return get_input_tensors(model, df)
    finally:
        model.model.cpu()


@ray.remote(max_calls=1)
def get_total_attribution_task(
    model: LudwigModel,
    target_feature_name: str,
    target_indices: List[Optional[int]],
    inputs_encoded: List[Variable],
    baseline: List[Variable],
    use_global: bool,
    nsamples: int,
) -> List[np.array]:
    model.model.unskip()
    model.model.to(get_torch_device())
    try:
        return [
            get_total_attribution(
                model=model,
                target_feature_name=target_feature_name,
                target_idx=target_idx,
                feature_inputs=inputs_encoded,
                baseline=baseline,
                use_global=use_global,
                nsamples=nsamples,
            )
            for target_idx in tqdm(target_indices, desc="Explain")
        ]
    finally:
        model.model.cpu()


def split_list(v, n):
    """Splits a list into n roughly equal sub-lists.

    Source: https://stackoverflow.com/a/2135920
    """
    k, m = divmod(len(v), n)
    return (v[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))
