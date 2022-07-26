import pytest

from ludwig.constants import COMBINED, LOSS
from ludwig.features.category_feature import CategoryOutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.utils import trainer_utils
from ludwig.utils.metric_utils import TrainerMetric


def test_progress_tracker_empty():
    output_features = LudwigFeatureDict()
    output_features["category_feature"] = CategoryOutputFeature(
        {
            "name": "category_feature",
            "decoder": {
                "input_size": 10,
                "num_classes": 3,
            },
        },
        {},
    )

    progress_tracker = trainer_utils.get_new_progress_tracker(
        batch_size=5,
        best_eval_metric=0,
        best_reduce_learning_rate_eval_metric=0,
        best_increase_batch_size_eval_metric=0,
        learning_rate=0.01,
        output_features=output_features,
    )

    assert progress_tracker.log_metrics() == {
        "batch_size": 5,
        "best_valid_metric": 0,
        "epoch": 0,
        "last_improvement_steps": 0,
        "learning_rate": 0.01,
        "num_increases_bs": 0,
        "num_reductions_lr": 0,
        "steps": 0,
        "tune_checkpoint_num": 0,
    }


def test_progress_tracker():
    output_features = LudwigFeatureDict()
    output_features["category_feature"] = CategoryOutputFeature(
        {
            "name": "category_feature",
            "decoder": {
                "input_size": 10,
                "num_classes": 3,
            },
        },
        {},
    )

    progress_tracker = trainer_utils.get_new_progress_tracker(
        batch_size=5,
        best_eval_metric=0,
        best_reduce_learning_rate_eval_metric=0,
        best_increase_batch_size_eval_metric=0,
        learning_rate=0.01,
        output_features=output_features,
    )

    progress_tracker.validation_metrics[COMBINED][LOSS].append(TrainerMetric(epoch=1, step=10, value=0.1))
    progress_tracker.validation_metrics[COMBINED][LOSS].append(TrainerMetric(epoch=1, step=20, value=0.2))

    assert progress_tracker.log_metrics() == {
        "batch_size": 5,
        "best_valid_metric": 0,
        "epoch": 0,
        "last_improvement_steps": 0,
        "learning_rate": 0.01,
        "num_increases_bs": 0,
        "num_reductions_lr": 0,
        "steps": 0,
        "tune_checkpoint_num": 0,
        "validation_metrics.combined.loss": 0.2,
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
