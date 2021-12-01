from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.metric import Metric


def sequence_mask(lengths: Tensor, maxlen: Optional[int] = None, dtype=torch.bool):
    """Implements tf.sequence_mask in torch.

    From https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/2.
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    return mask.type(dtype)


def dynamic_partition(data: Tensor, partitions: Tensor, num_partitions: int):
    """Implements tf.dynamic_repartition in torch.

    From https://discuss.pytorch.org/t/equivalent-of-tf-dynamic-partition/53735.
    """
    res = []
    for i in range(num_partitions):
        res += [data[(partitions == i).nonzero().squeeze(1)]]
    return res


def masked_correct_predictions(targets: Tensor, preds: Tensor, targets_sequence_lengths: Tensor) -> Tensor:
    """
    Params:
        targets: 2D tensor
        preds: 2D tensor
    """
    truncated_preds = preds[:, : targets.shape[1]]
    padded_truncated_preds = F.pad(truncated_preds, pad=[0, targets.shape[1] - truncated_preds.shape[1]])
    correct_preds = padded_truncated_preds == targets

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
