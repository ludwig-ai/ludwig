from collections import defaultdict, namedtuple
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torchmetrics.metric import Metric

from ludwig.constants import COMBINED, LOSS, NAME, TYPE
from ludwig.modules.metric_registry import get_metric_names_for_type
from ludwig.types import FeatureConfigDict


def sequence_mask(lengths: Tensor, maxlen: Optional[int] = None, dtype=torch.bool) -> Tensor:
    """Implements tf.sequence_mask in torch.

    From https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/2.
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    return mask.type(dtype)


def dynamic_partition(data: Tensor, partitions: Tensor, num_partitions: int) -> List[Tensor]:
    """Implements tf.dynamic_partition in torch.

    From https://discuss.pytorch.org/t/equivalent-of-tf-dynamic-partition/53735.
    """
    assert data.size() == partitions.size()

    # Flatten data into 1D vectors to do partitioning correctly.
    data = data.view(-1)
    partitions = partitions.view(-1)
    result = []
    for i in range(num_partitions):
        result += [data[(partitions == i).nonzero().squeeze(1)]]
    return result


def masked_correct_predictions(targets: Tensor, preds: Tensor, targets_sequence_lengths: Tensor) -> Tensor:
    """Masks out special symbols, and returns tensor of correct predictions.

    Args:
        targets: 2D tensor [batch_size, sequence_length]
        preds: 2D tensor [batch_size, sequence_length]

    Returns:
        1D tensor of all correct predictions.
    """
    correct_preds = preds == targets

    mask = sequence_mask(lengths=targets_sequence_lengths, maxlen=correct_preds.shape[1], dtype=torch.int32)
    _, masked_correct_preds = dynamic_partition(data=correct_preds, partitions=mask, num_partitions=2)

    return masked_correct_preds.type(torch.float32)


def get_scalar_from_ludwig_metric(metric: Metric) -> float:
    """Returns the scalar value of a Ludwig metric.

    Params:
        metric: Metric object

    Returns:
        float: scalar value of the metric
    """
    return metric.compute().detach().cpu().numpy().item()


# Data for training and evaluation metrics.
TrainerMetric = namedtuple("TrainerMetric", ("epoch", "step", "value"))


def reduce_trainer_metrics_dict(
    dict_dict_trainer_metrics: Dict[str, Dict[str, List[TrainerMetric]]]
) -> Dict[str, Dict[str, List[float]]]:
    """Reduces Dict[feature_name, Dict[metric_name, List[TrainerMetric]]] to Dict[feature_name, Dict[metric_name,
    List[float]]].

    Used for flattening the results returned by trainer.py::train(), which come from ProgressTracker.
    """
    flattened_dict = defaultdict(lambda: defaultdict(list))
    for feature_name, trainer_metric_dict in dict_dict_trainer_metrics.items():
        for metric_name, trainer_metrics in trainer_metric_dict.items():
            for trainer_metric in trainer_metrics:
                flattened_dict[feature_name][metric_name].append(trainer_metric[-1])
    # Convert defaultdict to dict so JSON serialization works with dataclasses.asdict().
    return {k: dict(v) for k, v in flattened_dict.items()}


def get_metric_names(output_features: Dict[str, "OutputFeature"]) -> Dict[str, List[str]]:  # noqa
    """Returns a dict of output_feature_name -> list of metric names."""
    metrics_names = {}
    for output_feature_name, output_feature in output_features.items():
        metrics_names[output_feature_name] = sorted(list(get_metric_names_for_type(output_feature.type())))
    # Add combined loss.
    metrics_names[COMBINED] = [LOSS]
    return metrics_names


def get_feature_to_metric_names_map(output_features: List[FeatureConfigDict]) -> Dict[str, List[str]]:
    """Returns a dict of output_feature_name -> list of metric names."""
    metrics_names = {}
    for output_feature in output_features:
        output_feature_name = output_feature[NAME]
        output_feature_type = output_feature[TYPE]
        metrics_names[output_feature_name] = get_metric_names_for_type(output_feature_type)
    metrics_names[COMBINED] = [LOSS]
    return metrics_names


def get_feature_to_metric_names_map_from_feature_collection(
    output_features: "FeatureCollection",  # noqa
) -> Dict[str, List[str]]:
    """Returns a dict of output_feature_name -> list of metric names."""
    metrics_names = {
        output_feature.name: get_metric_names_for_type(output_feature.type) for output_feature in output_features
    }
    metrics_names[COMBINED] = [LOSS]
    return metrics_names
