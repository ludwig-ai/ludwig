import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ludwig.schema.lr_scheduler import LRSchedulerConfig


class LRScheduler:
    def __init__(self, config: LRSchedulerConfig, optimizer: Optimizer):
        self.config = config
        self.optimizer = optimizer

        # Scheduler updated each training step
        self._train_scheduler = None

        # Scheduler updated each eval step
        self._eval_scheduler = None

    def reset(self, steps_per_checkpoint: int, total_steps: int):
        self._train_scheduler = get_linear_schedule_with_warmup(
            self.config, self.optimizer, steps_per_checkpoint, total_steps
        )

    def step(self):
        pass


def get_linear_schedule_with_warmup(
    config: LRSchedulerConfig,
    optimizer: Optimizer,
    steps_per_checkpoint: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if config.learning_rate_warmup_fraction > 0 and config.learning_rate_warmup_evaluations > 0:
        raise ValueError(
            f"Cannot specify both learning_rate_warmup_fraction ({config.learning_rate_warmup_fraction}) and "
            f"learning_rate_warmup_evaluations ({config.learning_rate_warmup_evaluations}) in the same config."
        )

    num_warmup_steps = 0
    if config.learning_rate_warmup_fraction > 0:
        num_warmup_steps = config.learning_rate_warmup_fraction * num_training_steps
    elif config.learning_rate_warmup_evaluations > 0:
        num_warmup_steps = config.learning_rate_warmup_evaluations * steps_per_checkpoint

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def exponential_decay(initial_learning_rate, decay_rate, decay_steps, step, staircase=False):
    decay_rate = float(decay_rate)
    decay_steps = float(decay_steps)
    step = float(step)
    exponent = 1 + step / decay_steps
    if staircase:
        exponent = math.ceil(exponent)
    return initial_learning_rate * math.pow(decay_rate, exponent)
