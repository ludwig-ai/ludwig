from collections import OrderedDict

import torch

from ludwig.utils import metric_utils
from ludwig.utils.metric_utils import TrainerMetric


def test_dynamic_partition():
    data = torch.Tensor([10, 20, 30, 40, 50])
    partitions = torch.Tensor([0, 0, 1, 1, 0])

    partitioned_data = metric_utils.dynamic_partition(data, partitions, 2)

    assert torch.equal(partitioned_data[0], torch.Tensor([10.0, 20.0, 50.0]))
    assert torch.equal(partitioned_data[1], torch.Tensor([30.0, 40.0]))


def test_dynamic_partition_2D():
    data = torch.Tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
        ]
    )
    partitions = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0]])

    partitioned_data = metric_utils.dynamic_partition(data, partitions, 2)

    assert torch.equal(partitioned_data[0], torch.Tensor([9, 18]))
    assert torch.equal(
        partitioned_data[1],
        torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]),
    )


def test_masked_correct_predictions():
    preds = torch.tensor([[1, 5, 1, 5, 1, 5, 12, 12, 12], [10, 1, 5, 1, 5, 12, 12, 12, 12]])
    targets = torch.tensor([[1, 9, 5, 7, 5, 9, 13, 6, 0], [1, 9, 7, 13, 4, 7, 7, 7, 0]])
    targets_sequence_length = torch.tensor([8, 8])

    result = metric_utils.masked_correct_predictions(targets, preds, targets_sequence_length)

    assert torch.equal(
        result, torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_reduce_trainer_metrics_dict():
    dict_dict_trainer_metrics = {
        "feature_name": {"metric_name": [metric_utils.TrainerMetric(epoch=1, step=2, value=10)]}
    }

    result = metric_utils.reduce_trainer_metrics_dict(dict_dict_trainer_metrics)

    assert result == {"feature_name": {"metric_name": [10]}}


def test_reduce_trainer_metrics_dict_ordered_dict():
    dict_dict_trainer_metrics = OrderedDict(
        [
            (
                "category_5B6BF",
                OrderedDict(
                    [
                        ("loss", [TrainerMetric(epoch=0, step=1, value=0.0)]),
                        ("accuracy", [TrainerMetric(epoch=0, step=1, value=1.0)]),
                    ]
                ),
            ),
            ("combined", {"loss": [TrainerMetric(epoch=0, step=1, value=0.0)]}),
        ]
    )

    result = metric_utils.reduce_trainer_metrics_dict(dict_dict_trainer_metrics)

    assert result == {"category_5B6BF": {"accuracy": [1.0], "loss": [0.0]}, "combined": {"loss": [0.0]}}
