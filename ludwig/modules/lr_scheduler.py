import math
import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from ludwig.constants import MINIMIZE
from ludwig.modules.metric_modules import LudwigMetric, get_metric_cls

from ludwig.schema.lr_scheduler import LRSchedulerConfig
from ludwig.utils.metric_utils import get_scalar_from_ludwig_metric


class ReduceLROnPlateauLimited(ReduceLROnPlateau):
    def __init__(self, optimizer: Optimizer, mode: str, step_limit: int, factor: float, patience: int):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience)
        self.step_limit = step_limit
        self._steps = 0

    def step(self, metrics, epoch=None):
        if self._steps >= self.step_limit:
            return

        self._steps += 1
        return super().step(metrics, epich=epoch)


class LRScheduler:
    def __init__(self, config: LRSchedulerConfig, optimizer: Optimizer):
        self.config = config
        self.optimizer = optimizer
        self.validation_metric = get_metric_cls(self.config.reduce_eval_metric)

        # Scheduler updated each training step
        self._train_scheduler = None

        # Scheduler updated each eval step
        self._eval_scheduler = None

    def reset(self, steps_per_checkpoint: int, total_steps: int):
        self._train_scheduler = get_linear_schedule_with_warmup(
            self.config, self.optimizer, steps_per_checkpoint, total_steps
        )

        if self.config.reduce_on_plateau > 0:
            mode = "min" if self.validation_metric.get_objective() == MINIMIZE else "max"
            self._eval_scheduler = ReduceLROnPlateauLimited(
                optimizer=self.optimizer,
                mode=mode,
                step_limit=self.config.reduce_on_plateau,
                factor=self.config.reduce_on_plateau_rate,
                patience=self.config.reduce_on_plateau_patience,
            )

    def step(self):
        self._train_scheduler.step()

    def eval_step(self, validation_metric: LudwigMetric):
        self._eval_scheduler.step(get_scalar_from_ludwig_metric(validation_metric))


def get_linear_schedule_with_warmup(
    config: LRSchedulerConfig,
    optimizer: Optimizer,
    steps_per_checkpoint: int,
    num_training_steps: int,
) -> LambdaLR:
    """Creates a learning rate scheduler that updates each training step."""

    if config.warmup_fraction > 0 and config.warmup_evaluations > 0:
        logging.info(
            "Both `learning_rate_scheduler.warmup_fraction` and `learning_rate_scheduler.warmup_evaluations`. "
            "This will result in the greater of the two (as a function of the total training steps) being used."
        )

    num_warmup_steps = 0
    if config.warmup_fraction > 0:
        num_warmup_steps = max(config.warmup_fraction * num_training_steps, num_warmup_steps)
    if config.warmup_evaluations > 0:
        num_warmup_steps = max(config.warmup_evaluations * steps_per_checkpoint, num_warmup_steps)

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
