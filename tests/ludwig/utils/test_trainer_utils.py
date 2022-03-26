from ludwig.constants import COMBINED, LOSS
from ludwig.features.category_feature import CategoryOutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.utils import trainer_utils
from ludwig.utils.metric_utils import TrainerMetric


def test_progress_tracker_empty():
    output_features = LudwigFeatureDict()
    output_features["category_feature"] = CategoryOutputFeature(
        {"name": "category_feature", "input_size": 10, "num_classes": 3}, {}
    )

    progress_tracker = trainer_utils.ProgressTracker(
        batch_size=5,
        epoch=0,
        steps=0,
        last_improvement_steps=0,
        last_learning_rate_reduction_steps=0,
        last_increase_batch_size_steps=0,
        learning_rate=0.01,
        best_eval_metric=0,
        best_reduce_learning_rate_eval_metric=0,
        last_reduce_learning_rate_eval_metric_improvement=0,
        best_increase_batch_size_eval_metric=0,
        last_increase_batch_size_eval_metric_improvement=0,
        num_reductions_learning_rate=0,
        num_increases_batch_size=0,
        output_features=output_features,
        last_improvement=0,
        last_learning_rate_reduction=0,
        last_increase_batch_size=0,
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
    }


def test_progress_tracker():
    output_features = LudwigFeatureDict()
    output_features["category_feature"] = CategoryOutputFeature(
        {"name": "category_feature", "input_size": 10, "num_classes": 3}, {}
    )

    progress_tracker = trainer_utils.ProgressTracker(
        batch_size=5,
        epoch=0,
        steps=0,
        last_improvement_steps=0,
        last_learning_rate_reduction_steps=0,
        last_increase_batch_size_steps=0,
        learning_rate=0.01,
        best_eval_metric=0,
        best_reduce_learning_rate_eval_metric=0,
        last_reduce_learning_rate_eval_metric_improvement=0,
        best_increase_batch_size_eval_metric=0,
        last_increase_batch_size_eval_metric_improvement=0,
        num_reductions_learning_rate=0,
        num_increases_batch_size=0,
        output_features=output_features,
        last_improvement=0,
        last_learning_rate_reduction=0,
        last_increase_batch_size=0,
    )

    progress_tracker.vali_metrics[COMBINED][LOSS].append(TrainerMetric(epoch=1, step=10, value=0.1))
    progress_tracker.vali_metrics[COMBINED][LOSS].append(TrainerMetric(epoch=1, step=20, value=0.2))

    assert progress_tracker.log_metrics() == {
        "batch_size": 5,
        "best_valid_metric": 0,
        "epoch": 0,
        "last_improvement_steps": 0,
        "learning_rate": 0.01,
        "num_increases_bs": 0,
        "num_reductions_lr": 0,
        "steps": 0,
        "vali_metrics.combined.loss": 0.2,
    }
