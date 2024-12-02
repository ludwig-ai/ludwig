import math

import numpy as np
from torch.optim import SGD

from ludwig.features.number_feature import NumberInputFeature, NumberOutputFeature
from ludwig.modules.lr_scheduler import LRScheduler
from ludwig.schema.encoders.base import DenseEncoderConfig
from ludwig.schema.features.number_feature import ECDNumberOutputFeatureConfig, NumberInputFeatureConfig
from ludwig.schema.lr_scheduler import LRSchedulerConfig
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import get_new_progress_tracker


def test_lr_scheduler_warmup_decay():
    total_steps = 10000
    steps_per_checkpoint = 1000
    base_lr = 1.0
    warmup_fraction = 0.1

    module = NumberInputFeature(NumberInputFeatureConfig(name="num1", encoder=DenseEncoderConfig()))

    const_optimizer = SGD(module.parameters(), lr=base_lr)
    const_config = LRSchedulerConfig(warmup_evaluations=0)
    const_scheduler = LRScheduler(
        config=const_config,
        optimizer=const_optimizer,
        steps_per_checkpoint=steps_per_checkpoint,
        total_steps=total_steps,
    )

    linear_optimizer = SGD(module.parameters(), lr=base_lr)
    linear_config = LRSchedulerConfig(warmup_fraction=warmup_fraction, decay="linear")
    linear_scheduler = LRScheduler(
        config=linear_config,
        optimizer=linear_optimizer,
        steps_per_checkpoint=steps_per_checkpoint,
        total_steps=total_steps,
    )

    exp_optimizer = SGD(module.parameters(), lr=base_lr)
    exp_config = LRSchedulerConfig(warmup_fraction=warmup_fraction, decay="exponential")
    exp_scheduler = LRScheduler(
        config=exp_config, optimizer=exp_optimizer, steps_per_checkpoint=steps_per_checkpoint, total_steps=total_steps
    )

    cosine_optimizer = SGD(module.parameters(), lr=base_lr)
    cosine_config = LRSchedulerConfig(warmup_fraction=warmup_fraction, decay="cosine", t_0=steps_per_checkpoint)
    cosine_scheduler = LRScheduler(
        config=cosine_config,
        optimizer=cosine_optimizer,
        steps_per_checkpoint=steps_per_checkpoint,
        total_steps=total_steps,
    )

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

        cosine_scheduler.step()
        cosine_lr = cosine_optimizer.param_groups[0]["lr"]

        if step < warmup_steps:
            assert linear_lr == exp_lr, f"step: {step}"
            assert linear_lr == cosine_lr, f"step: {step}"
            assert linear_lr < base_lr, f"step: {step}"
        elif step == warmup_steps:
            assert linear_lr == base_lr, f"step: {step}"
            assert cosine_lr == base_lr, f"step: {step}"
            assert exp_lr < base_lr, f"step: {step}"
        else:
            assert linear_lr < base_lr, f"step: {step}"
            assert exp_lr < base_lr, f"step: {step}"
            assert cosine_lr <= base_lr, f"step: {step}"

    assert linear_lr < exp_lr
    assert exp_lr < cosine_lr
    assert cosine_lr == base_lr


def test_lr_scheduler_reduce_on_plateau():
    total_eval_steps = 100
    base_lr = 1.0
    reduce_limit = 3

    module = NumberInputFeature(NumberInputFeatureConfig(name="num1", encoder=DenseEncoderConfig()))
    output1 = NumberOutputFeature(ECDNumberOutputFeatureConfig(name="output1", input_size=10), output_features={})

    optimizer = SGD(module.parameters(), lr=base_lr)
    config = LRSchedulerConfig(
        warmup_evaluations=0,
        decay=None,
        reduce_on_plateau=reduce_limit,
        reduce_on_plateau_patience=10,
        reduce_on_plateau_rate=0.1,
    )
    scheduler = LRScheduler(config=config, optimizer=optimizer, steps_per_checkpoint=0, total_steps=0)

    progress_tracker = get_new_progress_tracker(
        batch_size=64,
        best_eval_metric_value=float("inf"),
        best_increase_batch_size_eval_metric=float("inf"),
        learning_rate=base_lr,
        output_features={"output1": output1},
    )

    num_reductions = 0

    last_lr = optimizer.param_groups[0]["lr"]
    steps_to_plateau = 5
    loss = 10.0
    for epoch in range(total_eval_steps):
        for i in range(100):
            # Simulate batch-wise steps. If we make a mistake, then this will reset
            # the learning rate.
            scheduler.step()

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

    # 3 reductions that multiply by 0.1 each time
    assert np.isclose(lr, 0.001)


def test_lr_scheduler_cosine_decay_fixed_period():
    total_steps = 10000
    steps_per_checkpoint = 1000
    base_lr = 1.0

    module = NumberInputFeature(NumberInputFeatureConfig(name="num1", encoder=DenseEncoderConfig()))

    optimizer = SGD(module.parameters(), lr=base_lr)
    config = LRSchedulerConfig(decay="cosine", t_0=steps_per_checkpoint, decay_rate=0, reduce_on_plateau=0)
    scheduler = LRScheduler(config=config, optimizer=optimizer, steps_per_checkpoint=0, total_steps=0)

    curr_lr = base_lr
    prev_lr = base_lr
    num_restarts = 0
    for step in range(total_steps + 1):
        # Cosine annealing formula
        expected_lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step % steps_per_checkpoint) / steps_per_checkpoint))
        assert np.isclose(curr_lr, expected_lr), f"step: {step}"

        if prev_lr < curr_lr:
            # Since Cosine decay is periodic, we should see the learning rate
            # decrease and then increase again.
            num_restarts += 1

        prev_lr = curr_lr
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]

    assert num_restarts == 10, f"num_restarts: {num_restarts}"


def test_lr_scheduler_cosine_decay_increasing_period():
    total_steps = 20000
    steps_per_checkpoint = 1000
    base_lr = 1.0

    module = NumberInputFeature(NumberInputFeatureConfig(name="num1", encoder=DenseEncoderConfig()))

    optimizer = SGD(module.parameters(), lr=base_lr)
    config = LRSchedulerConfig(
        decay="cosine",
        t_0=steps_per_checkpoint,
        t_mult=2,
        decay_rate=0,
        reduce_on_plateau=0,
    )
    scheduler = LRScheduler(
        config=config, optimizer=optimizer, steps_per_checkpoint=steps_per_checkpoint, total_steps=total_steps
    )

    curr_lr = base_lr
    prev_lr = base_lr
    num_restarts = 0
    for _ in range(total_steps + 1):
        if prev_lr < curr_lr:
            # Since Cosine decay is periodic, we should see the learning rate
            # decrease and then increase again.
            num_restarts += 1

        prev_lr = curr_lr
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]

    # 1000, 3000, 6000, 12000, 24000 (but we stop at 20000)
    assert num_restarts == 4, f"num_restarts: {num_restarts}"


def test_lr_scheduler_save_load():
    steps_per_checkpoint = 10
    total_steps = 100
    base_lr = 1.0
    reduce_limit = 3

    module = NumberInputFeature(NumberInputFeatureConfig(name="num1", encoder=DenseEncoderConfig()))
    output1 = NumberOutputFeature(ECDNumberOutputFeatureConfig(name="output1", input_size=10), output_features={})

    optimizer = SGD(module.parameters(), lr=base_lr)
    config = LRSchedulerConfig(warmup_fraction=0.2, reduce_on_plateau=reduce_limit)
    scheduler = LRScheduler(
        config=config, optimizer=optimizer, steps_per_checkpoint=steps_per_checkpoint, total_steps=total_steps
    )

    progress_tracker = get_new_progress_tracker(
        batch_size=64,
        best_eval_metric_value=float("inf"),
        best_increase_batch_size_eval_metric=float("inf"),
        learning_rate=base_lr,
        output_features={"output1": output1},
    )

    for _ in range(10):
        scheduler.step()

    progress_tracker.train_metrics["output1"]["loss"].append(TrainerMetric(epoch=0, step=10, value=1.0))
    scheduler.eval_step(progress_tracker, "output1")

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    optimizer2 = SGD(module.parameters(), lr=base_lr)
    scheduler2 = LRScheduler(
        config=config, optimizer=optimizer2, steps_per_checkpoint=steps_per_checkpoint, total_steps=total_steps
    )

    # Important: state needs to be loaded after init of optimizer and scheduler, otherwise
    # it can override loaded state
    optimizer2.load_state_dict(optimizer_state)
    scheduler2.load_state_dict(scheduler_state)

    lr = optimizer.param_groups[0]["lr"]
    assert lr == optimizer2.param_groups[0]["lr"]
    assert scheduler.state_dict() == scheduler2.state_dict()

    for _ in range(10):
        scheduler.step()
        scheduler2.step()

    progress_tracker.train_metrics["output1"]["loss"].append(TrainerMetric(epoch=1, step=20, value=0.8))
    scheduler.eval_step(progress_tracker, "output1")
    scheduler2.eval_step(progress_tracker, "output1")

    assert lr != optimizer.param_groups[0]["lr"]
    assert optimizer.param_groups[0]["lr"] == optimizer2.param_groups[0]["lr"]
    assert scheduler.state_dict() == scheduler2.state_dict()
