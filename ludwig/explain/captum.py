import copy
import gc
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr._utils.input_layer_wrapper import InputIdentity
from torch.autograd import Variable
from tqdm import tqdm

from ludwig.api import LudwigModel
from ludwig.api_annotations import PublicAPI
from ludwig.constants import (
    BINARY,
    CATEGORY,
    DATE,
    IMAGE,
    INPUT_FEATURES,
    MINIMUM_BATCH_SIZE,
    NAME,
    NUMBER,
    PREPROCESSING,
    SEQUENCE,
    SET,
    TEXT,
    UNKNOWN_SYMBOL,
)
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.explain.explainer import Explainer
from ludwig.explain.explanation import ExplanationsResult
from ludwig.explain.util import get_pred_col, replace_layer_with_copy
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.ecd import ECD
from ludwig.utils.torch_utils import DEVICE

logger = logging.getLogger(__name__)

# These types as provided as integer values and passed through an embedding layer that breaks integrated gradients.
# As such, we need to take care to encode them before handing them to the explainer.
EMBEDDED_TYPES = {SEQUENCE, TEXT, CATEGORY, SET, DATE}


@dataclass
class ExplanationRunConfig:
    """Mutable state containing runtime configuration for explanation process.

    This is useful for updating the batch size used during explanation so it can be propagated across calls to
    `get_total_attribution`.
    """

    batch_size: int


def retry_with_halved_batch_size(run_config: ExplanationRunConfig):
    """Function wrapper that retries an fn with a halved batch size.

    We want to maintain as large of a batch size as possible to maximize throughput. However, calculating explanations
    requires significantly more memory, and the original batch sized used during training may be too large and cause a
    CUDA OOM error, for example, if using GPUs.

    Will raise an error if a non-OOM error is raised, or if the batch size is reduced below 1 and the fn still fails.
    """

    def retry_with_halved_batch_size_fn(fn):
        def retry_with_halved_batch_size_wrapper(*args, **kwargs):
            latest_error = None
            while run_config.batch_size >= MINIMUM_BATCH_SIZE:
                try:
                    return fn(*args, **kwargs)
                except RuntimeError as e:
                    latest_error = e
                    # PyTorch only generates Runtime errors for CUDA OOM.
                    gc.collect()
                    if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                        logger.exception(f"OOM at batch_size={run_config.batch_size}, halving and trying again")
                        run_config.batch_size //= 2
                    else:
                        # Not a CUDA error
                        raise

            raise RuntimeError(
                f"Ran into latest error {latest_error} during explanation. "
                "If a CUDA out of memory error, then the batch size could not be reduced any further."
            )

        return retry_with_halved_batch_size_wrapper

    return retry_with_halved_batch_size_fn


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
        self.input_maps = LudwigFeatureDict()
        self.input_maps.update(
            {
                arg_name: InputIdentity(arg_name)
                for arg_name in self.model.input_features.keys()
                if self.model.input_features.get(arg_name).type() not in EMBEDDED_TYPES
            }
        )

    def forward(self, *args):
        # Add back the dictionary structure so it conforms to ECD format.
        input_features: LudwigFeatureDict = self.model.input_features
        inputs = {
            # Send the input through the identity layer so that we can use the output of the layer for attribution.
            # Except for text/category features where we use the embedding layer for attribution.
            feat_name: feat_input
            if input_features.get(feat_name).type() in EMBEDDED_TYPES
            else self.input_maps.get(feat_name)(feat_input)
            for feat_name, feat_input in zip(input_features.keys(), args)
        }

        outputs = self.model(inputs)

        # At this point we only have the raw logits, but to make explainability work we need the probabilities
        # and predictions as well, so derive them.
        predictions = {}
        for of_name in self.model.output_features:
            predictions[of_name] = self.model.output_features.get(of_name).predictions(outputs, of_name)

        pred_t = get_pred_col(predictions, self.target)

        # If the target feature is a non-scalar type (vector, set, etc.), sum it to get a scalar value.
        # https://github.com/pytorch/captum/issues/377
        if len(pred_t.shape) > 1 and self.model.output_features.get(self.target).type() not in {
            CATEGORY,
            NUMBER,
            BINARY,
        }:
            pred_t = torch.sum(pred_t.reshape(pred_t.shape[0], -1), dim=1)

        return pred_t


@PublicAPI(stability="experimental")
class IntegratedGradientsExplainer(Explainer):
    def explain(self) -> ExplanationsResult:
        """Explain the model's predictions using Integrated Gradients.

        # Return

        :return: ExplanationsResult containing the explanations.
            `global_explanations`: (Explanation) Aggregate explanation for the entire input data.

            `row_explanations`: (List[Explanation]) A list of explanations, one for each row in the input data. Each
            explanation contains the integrated gradients for each label in the target feature's vocab with respect to
            each input feature.

            `expected_values`: (List[float]) of length [output feature cardinality] Average convergence delta for each
            label in the target feature's vocab.
        """

        # TODO(travis): add back skip encoders at the end in finally. Shouldn't be an issue in most cases as we
        # typically perform explanations on a loaded model and don't use it to predict afterwards.
        self.model.model.unskip()
        self.model.model.to(DEVICE)

        input_features: LudwigFeatureDict = self.model.model.input_features
        run_config = ExplanationRunConfig(batch_size=self.model.config_obj.trainer.batch_size)

        get_input_tensors_with_retry = retry_with_halved_batch_size(run_config)(get_input_tensors)
        get_total_attribution_with_retry = retry_with_halved_batch_size(run_config)(get_total_attribution)

        # Convert input data into embedding tensors from the output of the model encoders.
        inputs_encoded = get_input_tensors_with_retry(self.model, self.inputs_df, run_config)
        sample_encoded = get_input_tensors_with_retry(self.model, self.sample_df, run_config)
        baseline = get_baseline(self.model, sample_encoded)

        # Compute attribution for each possible output feature label separately.
        expected_values = []
        for target_idx in tqdm(range(self.vocab_size), desc="Explain"):
            total_attribution, feat_to_token_attributions, total_attribution_global = get_total_attribution_with_retry(
                self.model,
                self.target_feature_name,
                target_idx if self.is_category_target else None,
                inputs_encoded,
                baseline,
                len(self.inputs_df),
                run_config,
            )

            # Aggregate token attributions
            feat_to_token_attributions_global = {}
            for feat_name, token_attributions in feat_to_token_attributions.items():
                token_attributions_global = defaultdict(float)
                # sum attributions for each token
                for token, token_attribution in (ta for tas in token_attributions for ta in tas):
                    token_attributions_global[token] += abs(token_attribution)
                # divide by number of samples to get average attribution per token
                token_attributions_global = {
                    token: token_attribution / max(0, len(token_attributions))
                    for token, token_attribution in token_attributions_global.items()
                }
                # convert to list of tuples and sort by attribution
                token_attributions_global = sorted(token_attributions_global.items(), key=lambda x: x[1], reverse=True)
                # keep only top 100 tokens
                token_attributions_global = token_attributions_global[:100]
                feat_to_token_attributions_global[feat_name] = token_attributions_global

            self.global_explanation.add(
                input_features.keys(), total_attribution_global, feat_to_token_attributions_global
            )

            for i, (feature_attributions, explanation) in enumerate(zip(total_attribution, self.row_explanations)):
                # Add the feature attributions to the explanation object for this row.
                explanation.add(
                    input_features.keys(),
                    feature_attributions,
                    {k: v[i] for k, v in feat_to_token_attributions.items()},
                )

            # TODO(travis): for force plots, need something similar to SHAP E[X]
            expected_values.append(0.0)

            if self.is_binary_target:
                # For binary targets, we only need to compute attribution for the positive class (see below).
                break

        # For binary targets, add an extra attribution for the negative class (false).
        if self.is_binary_target:
            le_true = self.global_explanation.label_explanations[0]
            negated_attributions = le_true.to_array() * -1
            negated_token_attributions = {
                fa.feature_name: [(t, -a) for t, a in fa.token_attributions]
                for fa in le_true.feature_attributions
                if fa.token_attributions is not None
            }
            # Prepend the negative class to the list of label explanations.
            self.global_explanation.add(
                input_features.keys(), negated_attributions, negated_token_attributions, prepend=True
            )

            for explanation in self.row_explanations:
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

        return ExplanationsResult(self.global_explanation, self.row_explanations, expected_values)


def get_input_tensors(
    model: LudwigModel, input_set: pd.DataFrame, run_config: ExplanationRunConfig
) -> List[torch.Tensor]:
    """Convert the input data into a list of variables, one for each input feature.

    # Inputs

    :param model: The LudwigModel to use for encoding.
    :param input_set: The input data to encode of shape [batch size, num input features].

    # Return

    :return: A list of variables, one for each input feature. Shape of each variable is [batch size, embedding size].
    """
    # Ignore sample_ratio from the model config, since we want to explain all the data.
    sample_ratio_bak = model.config_obj.preprocessing.sample_ratio
    model.config_obj.preprocessing.sample_ratio = 1.0

    config = model.config_obj.to_dict()
    training_set_metadata = copy.deepcopy(model.training_set_metadata)
    for feature in config[INPUT_FEATURES]:
        preprocessing = training_set_metadata[feature[NAME]][PREPROCESSING]
        if preprocessing.get("cache_encoder_embeddings"):
            preprocessing["cache_encoder_embeddings"] = False

    # Convert raw input data into preprocessed tensor data
    dataset, _ = preprocess_for_prediction(
        config,
        dataset=input_set,
        training_set_metadata=training_set_metadata,
        data_format="auto",
        split="full",
        include_outputs=False,
        backend=model.backend,
        callbacks=model.callbacks,
    )

    # Restore sample_ratio
    model.config_obj.preprocessing.sample_ratio = sample_ratio_bak

    # Make sure the number of rows in the preprocessed dataset matches the number of rows in the input data
    assert (
        dataset.to_df().shape[0] == input_set.shape[0]
    ), f"Expected {input_set.shape[0]} rows in preprocessed dataset, but got {dataset.to_df().shape[0]}"

    # Convert dataset into a dict of tensors, and split each tensor into batches to control GPU memory usage
    inputs = {
        name: torch.from_numpy(dataset.dataset[feature.proc_column]).split(run_config.batch_size)
        for name, feature in model.model.input_features.items()
    }

    # Dict of lists to list of dicts
    input_batches = [dict(zip(inputs, t)) for t in zip(*inputs.values())]

    # List of dicts to dict of lists
    preproc_inputs = {k: torch.cat([d[k] for d in input_batches]) for k in input_batches[0]}

    data_to_predict = [v for _, v in preproc_inputs.items()]
    tensors = []
    for t in data_to_predict:
        # TODO(travis): Consider changing to `if not torch.is_floating_point(t.dtype)` to simplify, then handle bool
        # case in this block.
        if t.dtype == torch.int8 or t.dtype == torch.int16 or t.dtype == torch.int32 or t.dtype == torch.int64:
            # Don't wrap input into a variable if it's an integer type, since it will be used as an index into the
            # embedding table. We explain the output of the embedding table, not the input to the embedding table using
            # LayerIntegratedGradients.
            tensors.append(t)
        else:
            # Wrap input into a variable so torch will track the gradient and LayerIntegratedGradients can explain it.
            if t.dtype == torch.bool:
                t = t.to(torch.float32)
            tensors.append(Variable(t, requires_grad=True))

    return tensors


def get_baseline(model: LudwigModel, sample_encoded: List[Variable]) -> List[torch.Tensor]:
    # TODO(travis): pre-compute this during training from the full training dataset.
    input_features: LudwigFeatureDict = model.model.input_features

    baselines = []
    for sample_input, (name, feature) in zip(sample_encoded, input_features.items()):
        metadata = model.training_set_metadata[name]
        if feature.type() == TEXT:
            PAD_IND = metadata.get("pad_idx", metadata.get("word_pad_idx"))
            token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
            baseline = token_reference.generate_reference(sequence_length=sample_input.shape[1], device=DEVICE)
        elif feature.type() == CATEGORY:
            most_popular_token = max(metadata["str2freq"], key=metadata["str2freq"].get)
            most_popular_tok_idx = metadata["str2idx"].get(most_popular_token)

            # If an unknown is defined, use that as the baseline index, else use the most popular token
            baseline_tok_idx = metadata["str2idx"].get(UNKNOWN_SYMBOL, most_popular_tok_idx)
            baseline = torch.tensor(baseline_tok_idx, device=DEVICE)
        elif feature.type() == IMAGE:
            baseline = torch.zeros_like(sample_input[0], device=DEVICE)
        else:
            # For a robust baseline, we take the mean of all samples from the training data.
            baseline = torch.mean(sample_input.float(), dim=0)
        baselines.append(baseline.unsqueeze(0))

    return baselines


def get_total_attribution(
    model: LudwigModel,
    target_feature_name: str,
    target_idx: Optional[int],
    feature_inputs: List[Variable],
    baseline: List[torch.Tensor],
    nsamples: int,
    run_config: ExplanationRunConfig,
) -> Tuple[npt.NDArray[np.float64], Dict[str, List[List[Tuple[str, float]]]]]:
    """Compute the total attribution for each input feature for each row in the input data.

    Args:
        model: The Ludwig model to explain.
        target_feature_name: The name of the target feature to explain.
        target_idx: The index of the target feature label to explain if the target feature is a category.
        feature_inputs: The preprocessed input data as a list of tensors of length [num_features].
        baseline: The baseline input data as a list of tensors of length [num_features].
        nsamples: The total number of samples in the input data.

    Returns:
        The token-attribution pair for each token in the input feature for each row in the input data. The members of
        the output tuple are structured as follows:

        `total_attribution_rows`: (npt.NDArray[np.float64]) of shape [num_rows, num_features]
        The total attribution for each input feature for each row in the input data.

        `feat_to_token_attributions`: (Dict[str, List[List[Tuple[str, float]]]]) with values of shape
        [num_rows, seq_len, 2]

        `total_attribution_global`: (npt.NDArray[np.float64]) of shape [num_features]
        The attribution for each input feature aggregated across all input data.
    """
    input_features: LudwigFeatureDict = model.model.input_features

    # Configure the explainer, which includes wrapping the model so its interface conforms to
    # the format expected by Captum.
    model.model.zero_grad()
    explanation_model = WrapperModule(model.model, target_feature_name)

    layers = []
    for feat_name, feat in input_features.items():
        if feat.type() in EMBEDDED_TYPES:
            # Get embedding layer from encoder, which is the first child of the encoder.
            target_layer = feat.encoder_obj.get_embedding_layer()

            # If the current layer matches any layer in the list, make a deep copy of the layer.
            if len(layers) > 0 and any(target_layer == layer for layer in layers):
                # Replace the layer with a deep copy of the layer to ensure that the attributions unique for each input
                # feature that uses a shared layer.
                # Recommended here: https://github.com/pytorch/captum/issues/794#issuecomment-1093021638
                replace_layer_with_copy(feat, target_layer)
                target_layer = feat.encoder_obj.get_embedding_layer()  # get the new copy
        else:
            # Get the wrapped input layer.
            target_layer = explanation_model.input_maps.get(feat_name)

        layers.append(target_layer)

    explainer = LayerIntegratedGradients(explanation_model, layers)

    feature_inputs_splits = [ipt.split(run_config.batch_size) for ipt in feature_inputs]
    baseline = [t.to(DEVICE) for t in baseline]

    total_attribution_rows = None
    total_attribution_global = None
    feat_to_token_attributions = defaultdict(list)
    for input_batch in zip(*feature_inputs_splits):
        input_batch = [ipt.to(DEVICE) for ipt in input_batch]
        attribution = explainer.attribute(
            tuple(input_batch),
            baselines=tuple(baseline),
            target=target_idx,
            # https://captum.ai/docs/faq#i-am-facing-out-of-memory-oom-errors-when-using-captum-how-do-i-resolve-this
            internal_batch_size=run_config.batch_size,
        )

        attributions_reduced = []
        for a in attribution:
            a_reduced = a.detach().cpu()
            if a_reduced.ndim == 2 or a_reduced.ndim == 3:
                # Reduces category-level attributions of shape [batch_size, embedding_dim] by summing over the
                # embedding dimension to get attributions of shape [batch_size].
                # Reduces token-level attributions of shape [batch_size, sequence_length, embedding_dim] by summing
                # over the embedding dimension to get attributions of shape [batch_size, sequence_length]. We keep
                # the sequence dimension so we can map the attributions to the tokens.
                a_reduced = a_reduced.sum(dim=-1)
            elif a_reduced.ndim == 4:
                # Reduce pixel-level attributions of shape [batch_size, num_channels, height, width] by summing
                # over the channel and spatial dimensions to get attributions of shape [batch_size].
                a_reduced = a_reduced.sum(dim=(1, 2, 3))
            attributions_reduced.append(a_reduced)

        for inputs, attrs, (name, feat) in zip(input_batch, attributions_reduced, input_features.items()):
            if feat.type() == TEXT:
                tok_attrs = get_token_attributions(model, name, inputs.detach().cpu(), attrs)
                feat_to_token_attributions[name].append(tok_attrs)

        # Reduce attribution to [num_input_features, batch_size] by summing over the sequence dimension (if present).
        attribution = [a.sum(dim=-1) if a.ndim == 2 else a for a in attributions_reduced]
        attribution = np.stack(attribution)

        # Transpose to [batch_size, num_input_features]
        attribution = attribution.T

        if total_attribution_rows is not None:
            total_attribution_rows = np.concatenate([total_attribution_rows, attribution], axis=0)
        else:
            total_attribution_rows = attribution

        if total_attribution_global is not None:
            total_attribution_global += attribution.sum(axis=0)
        else:
            total_attribution_global = attribution.sum(axis=0)

    total_attribution_global /= nsamples

    feat_to_token_attributions = {k: [e for lst in v for e in lst] for k, v in feat_to_token_attributions.items()}

    return total_attribution_rows, feat_to_token_attributions, total_attribution_global


def get_token_attributions(
    model: LudwigModel,
    feature_name: str,
    input_ids: torch.Tensor,
    token_attributions: torch.Tensor,
) -> List[List[Tuple[str, float]]]:
    """Convert token-level attributions to an array of token-attribution pairs of shape.

    [batch_size, sequence_length, 2].

    Args:
        model: The LudwigModel used to generate the attributions.
        feature_name: The name of the feature for which the attributions were generated.
        input_ids: The input ids of shape [batch_size, sequence_length].
        token_attributions: The token-level attributions of shape [batch_size, sequence_length].

    Returns:
        An array of token-attribution pairs of shape [batch_size, sequence_length, 2].
    """
    assert (
        input_ids.dtype == torch.int8
        or input_ids.dtype == torch.int16
        or input_ids.dtype == torch.int32
        or input_ids.dtype == torch.int64
    )

    # Normalize token-level attributions to visualize the relative importance of each token.
    norm = torch.linalg.norm(token_attributions, dim=1)
    # Safe divide by zero by setting the norm to 1 if the norm is 0.
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    token_attributions = token_attributions / norm.unsqueeze(-1)

    # map input ids to input tokens via the vocabulary
    feature = model.training_set_metadata[feature_name]
    vocab = feature.get("idx2str", feature.get("word_idx2str"))
    idx2str = np.vectorize(lambda idx: vocab[idx])
    input_tokens = idx2str(input_ids)

    # add attribution to the input tokens
    tok_attrs = [
        list(zip(t, a)) for t, a in zip(input_tokens, token_attributions.tolist())
    ]  # [batch_size, sequence_length, 2]

    return tok_attrs
