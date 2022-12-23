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
        self._train_scheduler.step()

    def eval_step(self):
        pass


def get_linear_schedule_with_warmup(
    config: LRSchedulerConfig,
    optimizer: Optimizer,
    steps_per_checkpoint: int,
    num_training_steps: int,
) -> LambdaLR:
    """Creates a learning rate scheduler that updates each training step."""

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

    decay_fn = decay_registry[config.decay]

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return decay_fn(current_step, num_training_steps, num_warmup_steps, config)

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def no_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    return 1.0


def linear_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def exponential_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    decay_rate = float(config.decay_rate)
    decay_steps = float(config.decay_steps)
    step = float(current_step)
    exponent = 1 + step / decay_steps
    if config.staircase:
        exponent = math.ceil(exponent)
    return math.pow(decay_rate, exponent)


decay_registry = {
    None: no_decay,
    "linear": linear_decay,
    "exponential": exponential_decay,
}
