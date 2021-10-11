from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_mask(
        lengths: Tensor,
        maxlen: Optional[int] = None,
        dtype=torch.bool
):
    """ Implements tf.sequence_mask in torch

    From https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/2.
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    return mask.type(dtype)


def dynamic_partition(data: Tensor, partitions: Tensor, num_partitions: int):
    """ Implements tf.dynamic_repartition in torch

    From https://discuss.pytorch.org/t/equivalent-of-tf-dynamic-partition/53735.
    """
    res = []
    for i in range(num_partitions):
        res += [data[(partitions == i).nonzero().squeeze(1)]]
    return res


def masked_corrected_predictions(
        targets: Tensor,
        preds: Tensor,
        targets_sequence_lengths: Tensor
) -> Tensor:
    """
    Params:
        targets: 2D tensor
        preds: 2D tensor
    """
    truncated_preds = preds[:, :targets.shape[1]]
    padded_truncated_preds = F.pad(
        truncated_preds, pad=[0, targets.shape[1] - truncated_preds.shape[1]])
    correct_preds = padded_truncated_preds == targets

    mask = sequence_mask(
        lengths=targets_sequence_lengths,
        maxlen=correct_preds.shape[1],
        dtype=torch.int32
    )
    _, masked_correct_preds = dynamic_partition(
        data=correct_preds,
        partitions=mask,
        num_partitions=2)

    return masked_correct_preds.type(torch.float32)


# TODO(shreya): After sequence loss
# def masked_sequence_corrected_predictions(
#         targets, predictions, targets_sequence_lengths
# ):
#     truncated_preds = predictions[:, : targets.shape[1]]
#     paddings = tf.stack(
#         [[0, 0], [0, tf.shape(targets)[1] - tf.shape(truncated_preds)[1]]]
#     )
#     padded_truncated_preds = tf.pad(truncated_preds, paddings, name="ptp")
#
#     correct_preds = tf.equal(padded_truncated_preds, targets)
#
#     mask = tf.sequence_mask(
#         targets_sequence_lengths, maxlen=correct_preds.shape[1], dtype=tf.int32
#     )
#
#     one_masked_correct_prediction = (
#         1.0
#         - tf.cast(mask, tf.float32)
#         + (tf.cast(mask, tf.float32) * tf.cast(correct_preds, tf.float32))
#     )
#     sequence_correct_preds = tf.reduce_prod(
#         one_masked_correct_prediction, axis=-1
#     )
#
#     return sequence_correct_preds


# TODO(shreya): No PyTorch CUDA implementation available
# def edit_distance(
#         targets, target_seq_length, predictions_sequence,
#         predictions_seq_length
# ):
#     predicts = to_sparse(
#         predictions_sequence,
#         predictions_seq_length,
#         tf.shape(predictions_sequence)[1],
#     )
#     labels = to_sparse(targets, target_seq_length, tf.shape(targets)[1])
#     edit_distance = tf.edit_distance(predicts, labels)
#     mean_edit_distance = tf.reduce_mean(edit_distance)
#     return edit_distance, mean_edit_distance
