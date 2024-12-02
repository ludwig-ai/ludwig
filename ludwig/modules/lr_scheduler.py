import logging
import math
from typing import Any, Callable, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, ReduceLROnPlateau, SequentialLR

from ludwig.constants import MINIMIZE, TRAINING, VALIDATION
from ludwig.modules.metric_registry import get_metric_objective
from ludwig.schema.lr_scheduler import LRSchedulerConfig
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import ProgressTracker

logger = logging.getLogger(__name__)


class ReduceLROnPLateauCappedDecreases(ReduceLROnPlateau):
    def __init__(self, optimizer: Optimizer, mode: str, reduce_limit: int, factor: float, patience: int):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience)
        self.reduce_limit = reduce_limit
        self._num_reduce_lr = 0

    def step(self, metrics):
        if self._num_reduce_lr >= self.reduce_limit:
            # Already reduced the LR as many times as we will allow
            return

        return super().step(metrics)

    @property
    def num_reduce_lr(self) -> int:
        return self._num_reduce_lr

    def _reduce_lr(self, epoch=None):
        """Overrides the base ReduceLROnPlateau implementation."""
        self._num_reduce_lr += 1
        self.apply_lr()

    def apply_lr(self):
        if self._num_reduce_lr == 0:
            return

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * math.pow(self.factor, self._num_reduce_lr), self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    logger.info(f"From ReduceLROnPLateauCappedDecreases, reducing learning rate to {new_lr}")


class LRScheduler:
    def __init__(
        self,
        config: LRSchedulerConfig,
        optimizer: Optimizer,
        steps_per_checkpoint: int,
        total_steps: int,
    ):
        self.config = config
        self.optimizer = optimizer

        # Scheduler updated each training step
        self.step_info = StepInfo(steps_per_checkpoint, total_steps, self.config)
        self._train_scheduler = get_schedule_with_warmup_and_decay(self.config, self.optimizer, self.step_info)

        # Scheduler updated each eval step
        self._eval_scheduler = None
        if self.config.reduce_on_plateau > 0:
            mode = "min" if get_metric_objective(self.config.reduce_eval_metric) == MINIMIZE else "max"
            self._eval_scheduler = ReduceLROnPLateauCappedDecreases(
                optimizer=self.optimizer,
                mode=mode,
                reduce_limit=self.config.reduce_on_plateau,
                factor=self.config.reduce_on_plateau_rate,
                patience=self.config.reduce_on_plateau_patience,
            )

    def step(self):
        """Called every step of training."""
        self._train_scheduler.step()

        if self._eval_scheduler is not None:
            # We apply this scheduler every eval step, not train step, so we don't want to call step() here.
            # However, we need to re-apply the LR reduction to the LR from the train scheduler, as the first scheduler
            # resets the LR back to the base LR.
            self._eval_scheduler.apply_lr()

    def eval_step(self, progress_tracker: ProgressTracker, validation_field: str):
        """Called every checkpoint evaluation step."""
        if self._eval_scheduler is None:
            # No reduce on plateau
            return

        if self.config.reduce_eval_split == TRAINING:
            split_metrics = progress_tracker.train_metrics
        elif self.config.reduce_eval_split == VALIDATION:
            split_metrics = progress_tracker.validation_metrics
        else:  # if self.config.reduce_eval_split == TEST:
            split_metrics = progress_tracker.test_metrics

        validation_metric = self.config.reduce_eval_metric
        last_metric: TrainerMetric = split_metrics[validation_field][validation_metric][-1]
        last_metric_value = last_metric[-1]

        prev_num_reductions = self._eval_scheduler.num_reduce_lr
        self._eval_scheduler.step(last_metric_value)

        num_reductions = self._eval_scheduler.num_reduce_lr
        if num_reductions > prev_num_reductions:
            # LR reduction -> update progress tracker
            progress_tracker.last_learning_rate_reduction_steps = progress_tracker.steps
            progress_tracker.last_learning_rate_reduction = 0
            progress_tracker.num_reductions_learning_rate += 1
        else:
            progress_tracker.last_learning_rate_reduction = (
                progress_tracker.steps - progress_tracker.last_learning_rate_reduction_steps
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "train_scheduler_state": self._train_scheduler.state_dict(),
            "eval_scheduler_state": self._eval_scheduler.state_dict() if self._eval_scheduler is not None else {},
        }

    def load_state_dict(self, d: Dict[str, Any]):
        self._train_scheduler.load_state_dict(d["train_scheduler_state"])
        if self._eval_scheduler is not None:
            self._eval_scheduler.load_state_dict(d["eval_scheduler_state"])


class StepInfo:
    """Stores the steps_per_checkpoint and total_steps used during the current training run.

    This class is needed by LambdaLR to allow us to update the steps on training init without resetting the entire
    LRScheduler from scratch (which would result in resetting the optimizer learning rate).
    """

    def __init__(self, steps_per_checkpoint: int, total_steps: int, config: LRSchedulerConfig):
        self.config = config
        self.steps_per_checkpoint = steps_per_checkpoint
        self.num_training_steps = total_steps

        if self.config.warmup_fraction > 0 and self.config.warmup_evaluations > 0:
            logger.info(
                "Both `learning_rate_scheduler.warmup_fraction` and `learning_rate_scheduler.warmup_evaluations` "
                "provided. The larger of the two (as a function of the total training steps) will be used."
            )

        num_warmup_steps = 0
        if self.config.warmup_fraction > 0:
            num_warmup_steps = max(self.config.warmup_fraction * self.num_training_steps, num_warmup_steps)
        if self.config.warmup_evaluations > 0:
            num_warmup_steps = max(self.config.warmup_evaluations * self.steps_per_checkpoint, num_warmup_steps)
        self.num_warmup_steps = num_warmup_steps


def get_schedule_with_warmup_and_decay(
    config: LRSchedulerConfig,
    optimizer: Optimizer,
    step_info: StepInfo,
) -> LambdaLR:
    """Creates a learning rate scheduler that updates each training step."""
    schedulers = []

    # Warmup scheduler.
    if step_info.num_warmup_steps > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lambda current_step: float(current_step) / float(max(1, step_info.num_warmup_steps)),
        )
        schedulers.append(warmup_scheduler)

    # Decay scheduler.
    decay = config.decay
    decay_scheduler = decay_registry[decay](config, optimizer, step_info)
    schedulers.append(decay_scheduler)

    if len(schedulers) == 1:
        # Only one scheduler, so no need to wrap in a SequentialLR.
        return schedulers[0]

    # Return a SequentialLR that applies the warmup and decay schedulers in order
    # with the warmup scheduler only applied for the first num_warmup_steps steps.
    return SequentialLR(optimizer, schedulers=schedulers, milestones=[step_info.num_warmup_steps])


def no_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    return 1.0


def linear_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    return max(
        0.0,
        float(num_training_steps - num_warmup_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def exponential_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    decay_rate = float(config.decay_rate)
    decay_steps = float(config.decay_steps)
    step = float(current_step)
    exponent = 1 + step / decay_steps
    if config.staircase:
        exponent = math.ceil(exponent)
    return math.pow(decay_rate, exponent)


def wrap_decay_fn(decay_fn: Callable) -> Callable:
    def init_fn(config: LRSchedulerConfig, optimizer: Optimizer, step_info: StepInfo) -> LambdaLR:
        return LambdaLR(
            optimizer,
            lambda current_step: decay_fn(
                current_step, step_info.num_training_steps, step_info.num_warmup_steps, config
            ),
        )

    return init_fn


def init_cosine_decay(
    config: LRSchedulerConfig,
    optimizer: Optimizer,
    step_info: StepInfo,
) -> CosineAnnealingWarmRestarts:
    t_0 = config.t_0
    if not t_0:
        t_0 = step_info.steps_per_checkpoint
    if not t_0:
        # A scheduler may be initialized with dummy values like at the start of training.
        # Ensure that t_0 != 0, as this causes an error to be raised.
        t_0 = 1

    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t_0,
        T_mult=config.t_mult or 1,
        eta_min=config.eta_min or 0,
    )


decay_registry = {
    None: wrap_decay_fn(no_decay),
    "linear": wrap_decay_fn(linear_decay),
    "exponential": wrap_decay_fn(exponential_decay),
    "cosine": init_cosine_decay,
}
