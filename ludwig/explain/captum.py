from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr._utils.input_layer_wrapper import InputIdentity
from torch.autograd import Variable
from tqdm import tqdm

from ludwig.api import LudwigModel
from ludwig.api_annotations import PublicAPI
from ludwig.constants import CATEGORY, TEXT, UNKNOWN_SYMBOL
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.explain.explainer import Explainer
from ludwig.explain.explanation import Explanation
from ludwig.explain.util import get_pred_col
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.ecd import ECD
from ludwig.utils.torch_utils import DEVICE


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
        self.input_maps = nn.ModuleDict(
            {
                arg_name: InputIdentity(arg_name)
                for arg_name in self.model.input_features.keys()
                if self.model.input_features[arg_name].type() not in {TEXT, CATEGORY}
            }
        )

    def forward(self, *args):
        # Add back the dictionary structure so it conforms to ECD format.
        inputs = {
            # Send the input through the identity layer so that we can use the output of the layer for attribution.
            # Except for text/category features where we use the embedding layer for attribution.
            feat_name: feat_input
            if self.model.input_features[feat_name].type() in {TEXT, CATEGORY}
            else self.input_maps[feat_name](feat_input)
            for feat_name, feat_input in zip(self.model.input_features.keys(), args)
        }

        outputs = self.model(inputs)

        # At this point we only have the raw logits, but to make explainability work we need the probabilities
        # and predictions as well, so derive them.
        predictions = {}
        for of_name in self.model.output_features:
            predictions[of_name] = self.model.output_features[of_name].predictions(outputs, of_name)

        return get_pred_col(predictions, self.target)


def get_input_tensors(model: LudwigModel, input_set: pd.DataFrame) -> List[torch.Tensor]:
    """Convert the input data into a list of variables, one for each input feature.

    # Inputs

    :param model: The LudwigModel to use for encoding.
    :param input_set: The input data to encode of shape [batch size, num input features].

    # Return

    :return: A list of variables, one for each input feature. Shape of each variable is [batch size, embedding size].
    """
    # Convert raw input data into preprocessed tensor data
    dataset, _ = preprocess_for_prediction(
        model.config_obj.to_dict(),
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
        name: torch.from_numpy(dataset.dataset[feature.proc_column]).split(model.config_obj.trainer.batch_size)
        for name, feature in model.model.input_features.items()
    }

    # Dict of lists to list of dicts
    input_batches = [dict(zip(inputs, t)) for t in zip(*inputs.values())]

    # List of dicts to dict of lists
    encoded_inputs = {k: torch.cat([d[k] for d in input_batches]) for k in input_batches[0]}

    data_to_predict = [v for _, v in encoded_inputs.items()]
    tensors = []
    for t in data_to_predict:
        if t.dtype == torch.int8 or t.dtype == torch.int16 or t.dtype == torch.int32 or t.dtype == torch.int64:
            # Don't wrap input into a variable if it's an integer type, since it will be used as an index into the
            # embedding table. We explain the output of the embedding table, not the input to the embedding table using
            # LayerIntegratedGradients.
            tensors.append(t)
        else:
            # Wrap input into a variable so torch will track the gradient and LayerIntegratedGradients can explain it.
            tensors.append(Variable(t, requires_grad=True))

    return tensors


@PublicAPI(stability="experimental")
class IntegratedGradientsExplainer(Explainer):
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
        inputs_encoded = get_input_tensors(self.model, self.inputs_df)
        sample_encoded = get_input_tensors(self.model, self.sample_df)
        baseline = get_baseline(self.model, sample_encoded)

        # Compute attribution for each possible output feature label separately.
        expected_values = []
        for target_idx in tqdm(range(self.vocab_size), desc="Explain"):
            total_attribution = get_total_attribution(
                self.model,
                self.target_feature_name,
                target_idx if self.is_category_target else None,
                inputs_encoded,
                baseline,
                self.use_global,
                len(self.inputs_df),
            )

            for feature_attributions, explanation in zip(total_attribution, self.explanations):
                # Add the feature attributions to the explanation object for this row.
                explanation.add(feature_attributions)

            # TODO(travis): for force plots, need something similar to SHAP E[X]
            expected_values.append(0.0)

            if self.is_binary_target:
                # For binary targets, we only need to compute attribution for the positive class (see below).
                break

        # For binary targets, add an extra attribution for the negative class (false).
        if self.is_binary_target:
            for explanation in self.explanations:
                le_true = explanation.label_explanations[0]
                explanation.add(le_true.feature_attributions * -1)

            # TODO(travis): for force plots, need something similar to SHAP E[X]
            expected_values.append(0.0)

        return self.explanations, expected_values


def get_baseline(model: LudwigModel, sample_encoded: List[Variable]) -> List[torch.Tensor]:
    # TODO(travis): pre-compute this during training from the full training dataset.
    input_features: LudwigFeatureDict = model.model.input_features

    baselines = []
    for sample_input, (name, feature) in zip(sample_encoded, input_features.items()):
        metadata = model.training_set_metadata[name]
        if feature.type() == TEXT:
            PAD_IND = metadata["pad_idx"]
            token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
            baseline = token_reference.generate_reference(sequence_length=sample_input.shape[1], device=DEVICE)
        elif feature.type() == CATEGORY:
            most_popular_token = max(metadata["str2freq"], key=metadata["str2freq"].get)
            most_popular_tok_idx = metadata["str2idx"].get(most_popular_token)

            # If an unknown is defined, use that as the baseline index, else use the most popular token
            baseline_tok_idx = metadata["str2idx"].get(UNKNOWN_SYMBOL, most_popular_tok_idx)
            baseline = torch.tensor(baseline_tok_idx, device=DEVICE)
        else:
            # For a robust baseline, we take the mean of all embeddings in the sample from the training data.
            # TODO(joppe): now that we don't have embeddings, we should re-evaluate this.
            baseline = torch.mean(sample_input.float(), dim=0)
        baselines.append(baseline.unsqueeze(0))

    return baselines


def get_total_attribution(
    model: LudwigModel,
    target_feature_name: str,
    target_idx: Optional[int],
    inputs_encoded: List[Variable],
    baseline: List[torch.Tensor],
    use_global: bool,
    nsamples: int,
) -> np.array:

    # Configure the explainer, which includes wrapping the model so its interface conforms to
    # the format expected by Captum.
    model.model.zero_grad()
    explanation_model = WrapperModule(model.model, target_feature_name)

    layers = []
    for feat_name, feat in model.model.input_features.items():
        if feat.type() in {TEXT, CATEGORY}:
            # Get embedding layer from encoder, which is the first child of the encoder.
            layers.append(next(feat.encoder_obj.children()))
        else:
            # Get the wrapped input layer.
            layers.append(explanation_model.input_maps[feat_name])

    explainer = LayerIntegratedGradients(explanation_model, layers)

    inputs_encoded_splits = [ipt.split(model.config_obj.trainer.batch_size) for ipt in inputs_encoded]
    baseline = [t.to(DEVICE) for t in baseline]

    total_attribution = None
    for input_batch in zip(*inputs_encoded_splits):
        input_batch = [ipt.to(DEVICE) for ipt in input_batch]
        attribution = explainer.attribute(
            tuple(input_batch),
            baselines=tuple(baseline),
            target=target_idx,
        )

        attributions_reduced = []
        for a in attribution:
            a_reduced = a.detach().cpu()
            if a.ndim > 1:
                # Convert to token-level attributions by summing over the embedding dimension.
                a_reduced = a.sum(dim=-1).squeeze(0)
            if a_reduced.ndim == 2:
                # Normalize token-level attributions of shape [batch_size, sequence_length] by dividing by the
                # norm of the sequence.
                a_reduced = a_reduced / torch.norm(a_reduced)
            attributions_reduced.append(a_reduced.numpy())

        token_attributions = {}
        for inputs, attrs, feat_name in zip(input_batch, attributions_reduced, model.model.input_features.keys()):
            if attrs.ndim == 2:
                token_attributions[feat_name] = get_token_attributions(model, feat_name, inputs, attrs)

        # TODO: refactor below

        # Transpose to [batch_size, num_input_features]
        attribution = attribution.T

        if total_attribution is not None:
            if use_global:
                total_attribution += attribution.sum(axis=0, keepdims=True)
            else:
                total_attribution = np.concatenate([total_attribution, attribution], axis=0)
        else:
            if use_global:
                total_attribution = attribution.sum(axis=0, keepdims=True)
            else:
                total_attribution = attribution

    if use_global:
        total_attribution /= nsamples

    return total_attribution


def get_token_attributions(
    model: LudwigModel,
    feature_name: str,
    input_ids: torch.Tensor,
    token_attributions: np.array,
) -> np.array:
    """
    Convert token-level attributions to an array of token-attribution pairs of shape [batch_size, sequence_length, 2].

    # Inputs

    model: LudwigModel: The LudwigModel used to generate the attributions.
    feature_name: str: The name of the feature for which the attributions were generated.
    input_ids: torch.Tensor: The input ids of shape [batch_size, sequence_length].
    token_attributions: torch.Tensor: The token-level attributions of shape [batch_size, sequence_length].

    # Returns

    np.array: An array of token-attribution pairs of shape [batch_size, sequence_length, 2].
    """
    assert (
        input_ids.dtype == torch.int8
        or input_ids.dtype == torch.int16
        or input_ids.dtype == torch.int32
        or input_ids.dtype == torch.int64
    )

    # map input ids to input tokens via the vocabulary
    vocab = model.training_set_metadata[feature_name]["idx2str"]
    idx2str = np.vectorize(lambda idx: vocab[idx])
    input_tokens = idx2str(input_ids)

    # add attribution to the input tokens
    tok_attrs = np.stack((input_tokens, token_attributions), axis=2)  # [batch_size, sequence_length, 2]

    return tok_attrs
