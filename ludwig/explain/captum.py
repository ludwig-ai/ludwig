from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torch.autograd import Variable

from ludwig.api import LudwigModel
from ludwig.constants import BINARY, CATEGORY, TYPE
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.explain.base import Explainer
from ludwig.explain.util import Explanation, get_pred_col
from ludwig.models.ecd import ECD
from ludwig.utils.torch_utils import get_torch_device

DEVICE = get_torch_device()


class WrapperModule(torch.nn.Module):
    """Model used by the explainer to generate predictions.

    Unlike Ludwig's ECD class, this wrapper takes individual args as inputs to the forward function. We derive the order
    of these args from the order of the input_feature keys in ECD, which is guaranteed to be consistent (Python
    dictionaries are ordered consistently), so we can map back to the input feature dictionary as a second step within
    this wrapper.
    """

    def __init__(self, model: ECD, target: str):
        super().__init__()
        self.model = model
        self.target = target

    def forward(self, *args):
        preds = self.predict_from_encoded(*args)
        return get_pred_col(preds, self.target).cpu()

    def predict_from_encoded(self, *args):
        # Add back the dictionary structure so it conforms to ECD format.
        encoded_inputs = {}
        for k, v in zip(self.model.input_features.keys(), args):
            encoded_inputs[k] = {"encoder_output": v.to(DEVICE)}

        # Run the combiner and decoder separately since we already encoded the input.
        combined_outputs = self.model.combine(encoded_inputs)
        outputs = self.model.decode(combined_outputs, None, None)

        # At this point we only have the raw logits, but to make explainability work we need the probabilities
        # and predictions as well, so derive them.
        predictions = {}
        for of_name in self.model.output_features:
            predictions[of_name] = self.model.output_features[of_name].predictions(outputs, of_name)
        return predictions


def get_input_tensors(model: LudwigModel, input_set: pd.DataFrame) -> List[Variable]:
    """Convert the input data into a list of variables, one for each input feature.

    # Inputs

    :param model: The LudwigModel to use for encoding.
    :param input_set: The input data to encode of shape [batch size, num input features].

    # Return

    :return: A list of variables, one for each input feature. Shape of each variable is [batch size, embedding size].
    """

    # Convert raw input data into preprocessed tensor data
    dataset, _ = preprocess_for_prediction(
        model.config,
        dataset=input_set,
        training_set_metadata=model.training_set_metadata,
        data_format="auto",
        split="full",
        include_outputs=False,
        backend=model.backend,
        callbacks=model.callbacks,
    )

    # Convert dataset into a dict of tensors, and split each tensor into batches to control GPU memory usage
    inputs = {
        name: torch.from_numpy(dataset.dataset[feature.proc_column]).split(model.config["trainer"]["batch_size"])
        for name, feature in model.model.input_features.items()
    }

    # Dict of lists to list of dicts
    input_batches = [dict(zip(inputs, t)) for t in zip(*inputs.values())]

    # Encode the inputs into embedding space. This is necessary to ensure differentiability. Otherwise, category
    # and other features that pass through an embedding will not be explainable via gradient based methods.
    output_batches = []
    for batch in input_batches:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        output = model.model.encode(batch)

        # Extract the output tensor, discarding additional state used for sequence decoding.
        output = {k: v["encoder_output"].detach().cpu() for k, v in output.items()}
        output_batches.append(output)

    # List of dicts to dict of lists
    encoded_inputs = {k: torch.cat([d[k] for d in output_batches]) for k in output_batches[0]}

    # Wrap the output into a variable so torch will track the gradient.
    # TODO(travis): this won't work for text decoders, but we don't support explanations for those yet
    data_to_predict = [v for _, v in encoded_inputs.items()]
    data_to_predict = [Variable(t, requires_grad=True) for t in data_to_predict]

    return data_to_predict


class IntegratedGradientsExplainer(Explainer):
    def explain(self, **kwargs) -> Tuple[List[Explanation], List[float]]:
        """Explain the model's predictions using Integrated Gradients.

        # Return

        :return: (Tuple[List[Explanation], List[float]]) `(explanations, expected_values)`
            `explanations`: (List[Explanation]) A list of explanations, one for each row in the input data. Each
            explanation contains the integrated gradients for each label in the target feature's vocab with respect to
            each input feature.

            `expected_values`: (List[float]) of length [output feature cardinality] Average convergence delta for each
            label in the target feature's vocab.
        """
        # Convert input data into embedding tensors from the output of the model encoders.
        inputs_encoded = get_input_tensors(self.model, self.inputs_df)
        sample_encoded = get_input_tensors(self.model, self.sample_df)

        # For a robust baseline, we take the mean of all embeddings in the sample from the training data.
        # TODO(travis): pre-compute this during training from the full training dataset.
        baseline = [torch.unsqueeze(torch.mean(t, dim=0), 0) for t in sample_encoded]

        # Configure the explainer, which includes wrapping the model so its interface conforms to
        # the format expected by Captum.
        explanation_model = WrapperModule(self.model.model, self.target_feature_name)
        explainer = IntegratedGradients(explanation_model)

        # Compute attribution for each possible output feature label separately.
        expected_values = []
        for target_idx in range(self.vocab_size):
            attribution, delta = explainer.attribute(
                tuple(inputs_encoded),
                baselines=tuple(baseline),
                target=target_idx if self.is_category_target else None,
                internal_batch_size=self.model.config["trainer"]["batch_size"],
                return_convergence_delta=True,
            )

            # Attribution over the feature embeddings returns a vector with the same dimensions of
            # shape [batch_size, embedding_size], so take the sum over this vector in order to return a single
            # floating point attribution value per input feature.
            attribution = np.array([t.detach().numpy().sum(1) for t in attribution])

            # Transpose to [batch_size, num_input_features]
            attribution = attribution.T

            for feature_attributions, explanation in zip(attribution, self.explanations):
                # Add the feature attributions to the explanation object for this row.
                explanation.add(feature_attributions)

            # The convergence delta is given per row, so take the mean to compute the
            # average delta for the feature.
            # TODO(travis): this isn't really the expected value as it is for shap, so
            #  find a better name.
            expected_value = delta.detach().numpy().mean()
            expected_values.append(expected_value)

            if self.is_binary_target:
                # For binary targets, we only need to compute attribution for the positive class (see below).
                break

        # For binary targets, add an extra attribution for the negative class (false).
        if self.is_binary_target:
            for explanation in self.explanations:
                le_true = explanation.label_explanations[0]
                explanation.add(le_true.feature_attributions * -1)

            expected_values.append(expected_values[0] * -1)

        return self.explanations, expected_values
