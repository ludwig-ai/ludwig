from collections import OrderedDict
from typing import Union

import pytest

from ludwig.constants import AUTO, BATCH_SIZE, COMBINED, LOSS
from ludwig.features.category_feature import CategoryOutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.schema.features.category_feature import ECDCategoryOutputFeatureConfig
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils import trainer_utils
from ludwig.utils.metric_utils import TrainerMetric


def test_get_latest_metrics_dict():
    progress_tracker_metrics = OrderedDict(
        [
            (
                "category_92E9E",
                OrderedDict(
                    [
                        (
                            "loss",
                            [
                                TrainerMetric(epoch=0, step=1, value=0.7929425835609436),
                                TrainerMetric(epoch=1, step=2, value=0.7906522750854492),
                            ],
                        ),
                        (
                            "accuracy",
                            [
                                TrainerMetric(epoch=0, step=1, value=0.4117647111415863),
                                TrainerMetric(epoch=1, step=2, value=0.4117647111415863),
                            ],
                        ),
                    ]
                ),
            ),
            (
                "combined",
                {
                    "loss": [
                        TrainerMetric(epoch=0, step=1, value=0.7929425835609436),
                        TrainerMetric(epoch=1, step=2, value=0.7906522750854492),
                    ]
                },
            ),
        ]
    )

    latest_metrics_dict = trainer_utils.get_latest_metrics_dict(progress_tracker_metrics)

    assert latest_metrics_dict == {
        "category_92E9E": {"accuracy": 0.4117647111415863, "loss": 0.7906522750854492},
        "combined": {"loss": 0.7906522750854492},
    }


def test_get_latest_metrics_dict_empty():
    progress_tracker_metrics = OrderedDict(
        [("category_F18D1", OrderedDict([("loss", []), ("accuracy", [])])), ("combined", {"loss": []})]
    )

    latest_metrics_dict = trainer_utils.get_latest_metrics_dict(progress_tracker_metrics)

    assert latest_metrics_dict == {}


def test_progress_tracker_empty():
    output_features = LudwigFeatureDict()
    category_feature, _ = load_config_with_kwargs(
        ECDCategoryOutputFeatureConfig,
        {
            "name": "category_feature",
            "type": "category",
            "decoder": {
                "type": "classifier",
            },
            "num_classes": 3,
            "input_size": 10,
        },
    )
    output_features.set("category_feature", CategoryOutputFeature(category_feature, {}))

    progress_tracker = trainer_utils.get_new_progress_tracker(
        batch_size=5,
        best_eval_metric_value=0,
        best_increase_batch_size_eval_metric=0,
        learning_rate=0.01,
        output_features=output_features,
    )

    assert progress_tracker.log_metrics() == {
        "batch_size": 5,
        "best_valid_metric": 0,
        "epoch": 0,
        "best_eval_metric_steps": 0,
        "learning_rate": 0.01,
        "num_increases_bs": 0,
        "num_reductions_lr": 0,
        "steps": 0,
        "tune_checkpoint_num": 0,
        "best_eval_metric_checkpoint_number": 0,
        "best_eval_metric_epoch": 0,
        "checkpoint_number": 0,
        "last_improvement_steps": 0,
    }


def test_progress_tracker():
    output_features = LudwigFeatureDict()
    category_feature, _ = load_config_with_kwargs(
        ECDCategoryOutputFeatureConfig,
        {
            "name": "category_feature",
            "type": "category",
            "decoder": {
                "type": "classifier",
            },
            "num_classes": 3,
            "input_size": 10,
        },
    )
    output_features.set("category_feature", CategoryOutputFeature(category_feature, {}))

    progress_tracker = trainer_utils.get_new_progress_tracker(
        batch_size=5,
        best_eval_metric_value=0,
        best_increase_batch_size_eval_metric=0,
        learning_rate=0.01,
        output_features=output_features,
    )

    progress_tracker.validation_metrics[COMBINED][LOSS].append(TrainerMetric(epoch=1, step=10, value=0.1))
    progress_tracker.validation_metrics[COMBINED][LOSS].append(TrainerMetric(epoch=1, step=20, value=0.2))

    assert progress_tracker.log_metrics() == {
        "batch_size": 5,
        "best_eval_metric_checkpoint_number": 0,
        "best_eval_metric_epoch": 0,
        "best_valid_metric": 0,
        "checkpoint_number": 0,
        "epoch": 0,
        "best_eval_metric_steps": 0,
        "learning_rate": 0.01,
        "num_increases_bs": 0,
        "num_reductions_lr": 0,
        "steps": 0,
        "tune_checkpoint_num": 0,
        "validation_metrics.combined.loss": 0.2,
        "last_improvement_steps": 0,
    }


def test_full_progress_tracker():
    progress_tracker = trainer_utils.ProgressTracker(
        **{
            BATCH_SIZE: 128,
            "best_eval_metric_checkpoint_number": 7,
            "best_eval_metric_epoch": 6,
            "best_eval_metric_steps": 35,
            "best_eval_metric_value": 0.719,
            "last_improvement_steps": 35,
            "best_eval_test_metrics": {
                "Survived": {"accuracy": 0.634, "loss": 3.820, "roc_auc": 0.598},
                "combined": {"loss": 3.820},
            },
            "best_eval_train_metrics": {
                "Survived": {"accuracy": 0.682, "loss": 4.006, "roc_auc": 0.634},
                "combined": {"loss": 4.006},
            },
            "best_eval_validation_metrics": {
                "Survived": {"accuracy": 0.719, "loss": 4.396, "roc_auc": 0.667},
                "combined": {"loss": 4.396},
            },
            "best_increase_batch_size_eval_metric": float("inf"),
            "checkpoint_number": 12,
            "epoch": 12,
            "last_increase_batch_size": 0,
            "last_increase_batch_size_eval_metric_improvement": 0,
            "last_increase_batch_size_steps": 0,
            "last_learning_rate_reduction": 0,
            "last_learning_rate_reduction_steps": 0,
            "learning_rate": 0.001,
            "num_increases_batch_size": 0,
            "num_reductions_learning_rate": 0,
            "steps": 60,
            "test_metrics": {
                "Survived": {
                    "accuracy": [
                        [0, 5, 0.651],
                        [1, 10, 0.651],
                    ],
                    "loss": [
                        [0, 5, 4.130],
                        [1, 10, 4.074],
                    ],
                    "roc_auc": [
                        [0, 5, 0.574],
                        [1, 10, 0.595],
                    ],
                },
                "combined": {
                    "loss": [
                        [0, 5, 4.130],
                        [1, 10, 4.074],
                    ]
                },
            },
            "train_metrics": {
                "Survived": {
                    "accuracy": [
                        [0, 5, 0.6875],
                        [1, 10, 0.6875],
                    ],
                    "loss": [
                        [0, 5, 4.417],
                        [1, 10, 4.344],
                    ],
                    "roc_auc": [
                        [0, 5, 0.628],
                        [1, 10, 0.629],
                    ],
                },
                "combined": {
                    "loss": [
                        [0, 5, 4.417],
                        [1, 10, 4.344],
                    ]
                },
            },
            "tune_checkpoint_num": 0,
            "validation_metrics": {
                "Survived": {
                    "accuracy": [
                        [0, 5, 0.696],
                        [1, 10, 0.696],
                    ],
                    "loss": [
                        [0, 5, 4.494],
                        [1, 10, 4.473],
                    ],
                    "roc_auc": [
                        [0, 5, 0.675],
                        [1, 10, 0.671],
                    ],
                },
                "combined": {
                    "loss": [
                        [0, 5, 4.494],
                        [1, 10, 4.473],
                    ]
                },
            },
        }
    )

    assert progress_tracker.log_metrics() == {
        BATCH_SIZE: 128,
        "best.train_metrics.Survived.accuracy": 0.682,
        "best.train_metrics.Survived.loss": 4.006,
        "best.train_metrics.Survived.roc_auc": 0.634,
        "best.train_metrics.combined.loss": 4.006,
        "best.test_metrics.Survived.accuracy": 0.634,
        "best.test_metrics.Survived.loss": 3.82,
        "best.test_metrics.Survived.roc_auc": 0.598,
        "best.test_metrics.combined.loss": 3.82,
        "best.validation_metrics.Survived.accuracy": 0.719,
        "best.validation_metrics.Survived.loss": 4.396,
        "best.validation_metrics.Survived.roc_auc": 0.667,
        "best.validation_metrics.combined.loss": 4.396,
        "best_eval_metric_checkpoint_number": 7,
        "best_eval_metric_epoch": 6,
        "best_eval_metric_steps": 35,
        "best_valid_metric": 0.719,
        "checkpoint_number": 12,
        "epoch": 12,
        "last_improvement_steps": 35,
        "learning_rate": 0.001,
        "num_increases_bs": 0,
        "num_reductions_lr": 0,
        "steps": 60,
        "test_metrics.Survived.accuracy": 0.651,
        "test_metrics.Survived.loss": 4.074,
        "test_metrics.Survived.roc_auc": 0.595,
        "test_metrics.combined.loss": 4.074,
        "train_metrics.Survived.accuracy": 0.6875,
        "train_metrics.Survived.loss": 4.344,
        "train_metrics.Survived.roc_auc": 0.629,
        "train_metrics.combined.loss": 4.344,
        "tune_checkpoint_num": 0,
        "validation_metrics.Survived.accuracy": 0.696,
        "validation_metrics.Survived.loss": 4.473,
        "validation_metrics.Survived.roc_auc": 0.671,
        "validation_metrics.combined.loss": 4.473,
    }


def test_get_final_steps_per_checkpoint():
    # steps_per_checkpoint and checkpoints_per_epoch cannot both be specified.
    with pytest.raises(Exception):
        trainer_utils.get_final_steps_per_checkpoint(
            steps_per_epoch=1024,
            steps_per_checkpoint=1,
            checkpoints_per_epoch=1,
        )

    assert trainer_utils.get_final_steps_per_checkpoint(steps_per_epoch=1024, steps_per_checkpoint=100) == 100
    assert trainer_utils.get_final_steps_per_checkpoint(steps_per_epoch=1024, steps_per_checkpoint=2048) == 1024
    assert trainer_utils.get_final_steps_per_checkpoint(steps_per_epoch=1024, checkpoints_per_epoch=2) == 512
    assert trainer_utils.get_final_steps_per_checkpoint(steps_per_epoch=1024, checkpoints_per_epoch=2.5) == 409
    assert trainer_utils.get_final_steps_per_checkpoint(steps_per_epoch=1024, checkpoints_per_epoch=0.5) == 1024
    assert trainer_utils.get_final_steps_per_checkpoint(steps_per_epoch=1024) == 1024
    assert (
        trainer_utils.get_final_steps_per_checkpoint(
            steps_per_epoch=1024, steps_per_checkpoint=0, checkpoints_per_epoch=0
        )
        == 1024
    )


@pytest.mark.parametrize(
    "effective_batch_size,batch_size,gradient_accumulation_steps,num_workers,expected_batch_size,expected_grad_accum",
    [
        (128, 16, 4, 2, 16, 4),
        (AUTO, 16, 4, 2, 16, 4),
        (128, 16, AUTO, 2, 16, 4),
        (128, AUTO, 4, 2, 16, 4),
        (128, AUTO, AUTO, 2, AUTO, AUTO),
        (AUTO, AUTO, AUTO, 2, AUTO, AUTO),
        (AUTO, 16, AUTO, 2, 16, 1),
        (AUTO, AUTO, 4, 2, AUTO, 4),
    ],
)
def test_get_rendered_batch_size_grad_accum(
    effective_batch_size: Union[str, int],
    batch_size: Union[str, int],
    gradient_accumulation_steps: Union[str, int],
    num_workers: int,
    expected_batch_size: int,
    expected_grad_accum: int,
):
    config = ECDTrainerConfig.from_dict(
        {
            "effective_batch_size": effective_batch_size,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
    )
    rendered_batch_size, rendered_grad_accum = trainer_utils.get_rendered_batch_size_grad_accum(config, num_workers)
    assert rendered_batch_size == expected_batch_size
    assert rendered_grad_accum == expected_grad_accum
