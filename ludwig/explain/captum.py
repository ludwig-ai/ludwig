from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torch.autograd import Variable

from ludwig.api import LudwigModel
from ludwig.constants import BINARY, CATEGORY, TYPE
from ludwig.data.dataset.ray import RayDataset
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.explain.util import get_feature_name, get_pred_col, prepare_data
from ludwig.models.ecd import ECD
from ludwig.utils.dataframe_utils import is_dask_object
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

    if isinstance(dataset, RayDataset):
        dataset_df = dataset.to_df()
        inputs = {}
        for name, feature in model.model.input_features.items():
            col = dataset_df[feature.proc_column]
            if is_dask_object(col, model.backend):
                col = col.compute()
            if isinstance(col, pd.Series):
                col = col.to_numpy()
            inputs[name] = torch.from_numpy(col).split(model.config["trainer"]["batch_size"])
    else:
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


def explain_ig(
    model: LudwigModel, inputs_df: pd.DataFrame, sample_df: pd.DataFrame, target: str
) -> Tuple[np.array, List[float], np.array]:
    model.model.to(DEVICE)

    inputs_df, sample_df, _, target_feature_name = prepare_data(model, inputs_df, sample_df, target)

    # Convert input data into embedding tensors from the output of the model encoders.
    inputs_encoded = get_input_tensors(model, inputs_df)
    sample_encoded = get_input_tensors(model, sample_df)

    # For a robust baseline, we take the mean of all embeddings in the sample from the training data.
    # TODO(travis): pre-compute this during training from the full training dataset.
    baseline = [torch.unsqueeze(torch.mean(t, dim=0), 0) for t in sample_encoded]

    # Configure the explainer, which includes wrapping the model so its interface conforms to
    # the format expected by Captum.
    target_feature_name = get_feature_name(model, target)
    explanation_model = WrapperModule(model.model, target_feature_name)
    explainer = IntegratedGradients(explanation_model)

    # Lookup from column name to output feature
    output_feature_map = {feature["column"]: feature for feature in model.config["output_features"]}

    # The second dimension of the attribution tensor corresponds to the cardinality
    # of the output feature. For regression (number) this is 1, for binary 2, and
    # for category it is the vocab size.
    vocab_size = 1
    is_category_target = output_feature_map[target_feature_name][TYPE] == CATEGORY
    if is_category_target:
        vocab_size = model.training_set_metadata[target_feature_name]["vocab_size"]

    # Compute attribution for each possible output feature label separately.
    attribution_by_label = []
    expected_values = []
    for target_idx in range(vocab_size):
        attribution, delta = explainer.attribute(
            tuple(inputs_encoded),
            baselines=tuple(baseline),
            target=target_idx if is_category_target else None,
            internal_batch_size=model.config["trainer"]["batch_size"],
            return_convergence_delta=True,
        )

        # Attribution over the feature embeddings returns a vector with the same
        # dimensions, so take the sum over this vector in order to return a single
        # floating point attribution value per input feature.
        attribution = np.array([t.detach().numpy().sum(1) for t in attribution])
        attribution_by_label.append(attribution)

        # The convergence delta is given per row, so take the mean to compute the
        # average delta for the feature.
        # TODO(travis): this isn't really the expected value as it is for shap, so
        #  find a better name.
        expected_value = delta.detach().numpy().mean()
        expected_values.append(expected_value)

    # For binary outputs, add an extra attribution for the negative class (false).
    is_binary_target = output_feature_map[target_feature_name][TYPE] == BINARY
    if is_binary_target:
        attribution_by_label.append(attribution_by_label[0] * -1)
        expected_values.append(expected_values[0] * -1)

    # Stack the attributions into a single tensor of shape:
    # [batch_size, output_feature_cardinality, num_input_features]
    attribution = np.stack(attribution_by_label, axis=1)
    attribution = np.transpose(attribution, (2, 1, 0))

    # Add in predictions as part of the result.
    pred_df = model.predict(inputs_df, return_type=dict)[0]
    preds = np.array(pred_df[target_feature_name]["predictions"])

    return attribution, expected_values, preds
