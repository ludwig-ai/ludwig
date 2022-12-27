from torch.optim import SGD

from ludwig.features.number_feature import NumberInputFeature, NumberOutputFeature
from ludwig.modules.lr_scheduler import LRScheduler
from ludwig.schema.decoders.base import PassthroughDecoderConfig
from ludwig.schema.encoders.base import DenseEncoderConfig
from ludwig.schema.features.number_feature import NumberInputFeatureConfig, NumberOutputFeatureConfig
from ludwig.schema.lr_scheduler import LRSchedulerConfig
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import ProgressTracker, get_new_progress_tracker


def test_lr_scheduler_warmup_decay():
    total_steps = 10000
    steps_per_checkpoint = 1000
    base_lr = 1.0
    warmup_fraction = 0.1

    module = NumberInputFeature(NumberInputFeatureConfig(name="num1", encoder=DenseEncoderConfig()))

    const_optimizer = SGD(module.parameters(), lr=base_lr)
    const_config = LRSchedulerConfig(warmup_evaluations=0)
    const_scheduler = LRScheduler(config=const_config, optimizer=const_optimizer)
    const_scheduler.reset(steps_per_checkpoint, total_steps)

    linear_optimizer = SGD(module.parameters(), lr=base_lr)
    linear_config = LRSchedulerConfig(warmup_fraction=warmup_fraction, decay="linear")
    linear_scheduler = LRScheduler(config=linear_config, optimizer=linear_optimizer)
    linear_scheduler.reset(steps_per_checkpoint, total_steps)

    exp_optimizer = SGD(module.parameters(), lr=base_lr)
    exp_config = LRSchedulerConfig(warmup_fraction=warmup_fraction, decay="exponential")
    exp_scheduler = LRScheduler(config=exp_config, optimizer=exp_optimizer)
    exp_scheduler.reset(steps_per_checkpoint, total_steps)

    warmup_steps = total_steps * warmup_fraction
    for i in range(total_steps):
        # Offset by 1
        step = i + 1

        const_scheduler.step()
        const_lr = const_optimizer.param_groups[0]["lr"]
        assert const_lr == base_lr, f"step: {step}"

        linear_scheduler.step()
        linear_lr = linear_optimizer.param_groups[0]["lr"]

        exp_scheduler.step()
        exp_lr = exp_optimizer.param_groups[0]["lr"]

        if step < warmup_steps:
            assert linear_lr == exp_lr, f"step: {step}"
            assert linear_lr < base_lr, f"step: {step}"
        elif step == warmup_steps:
            assert linear_lr == base_lr, f"step: {step}"
            assert exp_lr < base_lr, f"step: {step}"
        else:
            assert linear_lr < base_lr, f"step: {step}"
            assert exp_lr < base_lr, f"step: {step}"

    assert linear_lr < exp_lr


def test_lr_scheduler_reduce_on_plateau():
    total_eval_steps = 100
    base_lr = 1.0
    reduce_limit = 3

    module = NumberInputFeature(NumberInputFeatureConfig(name="num1", encoder=DenseEncoderConfig()))
    output1 = NumberOutputFeature(
        NumberOutputFeatureConfig(name="output1", input_size=10, decoder=PassthroughDecoderConfig()), output_features={}
    )

    optimizer = SGD(module.parameters(), lr=base_lr)
    config = LRSchedulerConfig(warmup_evaluations=0, reduce_on_plateau=reduce_limit)
    scheduler = LRScheduler(config=config, optimizer=optimizer)

    progress_tracker = get_new_progress_tracker(
        batch_size=64,
        best_eval_metric=float("inf"),
        best_increase_batch_size_eval_metric=float("inf"),
        learning_rate=base_lr,
        output_features={"output1": output1},
    )

    num_reductions = 0

    last_lr = optimizer.param_groups[0]["lr"]
    steps_to_plateau = 5
    loss = 10.0
    for epoch in range(total_eval_steps):
        steps_to_plateau -= 1
        if steps_to_plateau > 0:
            loss -= 0.1

        progress_tracker.train_metrics["output1"]["loss"].append(
            TrainerMetric(epoch=epoch, step=epoch * 100, value=loss)
        )
        scheduler.eval_step(progress_tracker, "output1")
        lr = optimizer.param_groups[0]["lr"]
        if lr < last_lr:
            # Reset steps to plateau
            steps_to_plateau = 5
            num_reductions += 1
        last_lr = lr

    assert num_reductions == reduce_limit
