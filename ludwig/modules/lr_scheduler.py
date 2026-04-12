import logging
import math
from collections.abc import Callable
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, OneCycleLR, ReduceLROnPlateau, SequentialLR

from ludwig.constants import MINIMIZE, TRAINING, VALIDATION
from ludwig.modules.metric_registry import get_metric_objective
from ludwig.schema.lr_scheduler import LRSchedulerConfig
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import ProgressTracker

logger = logging.getLogger(__name__)


class ReduceLROnPLateauCappedDecreases(ReduceLROnPlateau):
    """ReduceLROnPlateau with a cap on the number of allowed reductions.

    Use when: you want to reduce LR in response to plateaus, but want to prevent the LR
    from collapsing to zero over a very long training run.

    Trade-offs: Requires a validation metric to be tracked. Does not interact well with
    schedules that already decay LR aggressively (e.g., one_cycle).
    """

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

    def state_dict(self) -> dict[str, Any]:
        return {
            "train_scheduler_state": self._train_scheduler.state_dict(),
            "eval_scheduler_state": self._eval_scheduler.state_dict() if self._eval_scheduler is not None else {},
        }

    def load_state_dict(self, d: dict[str, Any]):
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
    decay = config.decay

    # OneCycleLR manages warmup and decay internally — skip the SequentialLR wrapper.
    if decay == "one_cycle":
        if step_info.num_warmup_steps > 0:
            logger.warning(
                "decay='one_cycle' includes its own warmup phase controlled by `pct_start`. "
                "The `warmup_fraction`/`warmup_evaluations` settings will be ignored."
            )
        return init_one_cycle(config, optimizer, step_info)

    # WSD manages its own warmup internally via wsd_warmup_fraction.
    if decay == "wsd" and step_info.num_warmup_steps > 0:
        logger.warning(
            "decay='wsd' includes its own warmup phase controlled by `wsd_warmup_fraction`. "
            "The external `warmup_fraction`/`warmup_evaluations` settings will be ignored for WSD."
        )

    schedulers = []

    # Warmup scheduler.
    if step_info.num_warmup_steps > 0 and decay != "wsd":
        warmup_scheduler = LambdaLR(
            optimizer,
            lambda current_step: float(current_step) / float(max(1, step_info.num_warmup_steps)),
        )
        schedulers.append(warmup_scheduler)

    # Decay scheduler.
    decay_scheduler = decay_registry[decay](config, optimizer, step_info)
    schedulers.append(decay_scheduler)

    if len(schedulers) == 1:
        # Only one scheduler, so no need to wrap in a SequentialLR.
        return schedulers[0]

    # Return a SequentialLR that applies the warmup and decay schedulers in order
    # with the warmup scheduler only applied for the first num_warmup_steps steps.
    return SequentialLR(optimizer, schedulers=schedulers, milestones=[step_info.num_warmup_steps])


# ---------------------------------------------------------------------------
# Decay functions (used via wrap_decay_fn -> LambdaLR)
# ---------------------------------------------------------------------------


def no_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    """No decay: keep LR constant throughout training.

    Use when: you want full control via the optimizer's initial LR, or when using
    reduce_on_plateau alone.

    Trade-offs: Simple but rarely optimal for long runs; the LR never adapts to the loss
    landscape.
    """
    return 1.0


def linear_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    """Linear decay from base LR to 0 over the remaining training steps after warmup.

    Use when: fine-tuning pretrained models (catastrophic forgetting risk), or when a
    predictable, monotone LR reduction is preferred. Popular for BERT-style fine-tuning.

    Trade-offs: Aggressive and inflexible — once the schedule starts, LR can only decrease.
    May converge too fast if training is short.
    """
    return max(
        0.0,
        float(num_training_steps - num_warmup_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def exponential_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    """Exponential decay: lr *= decay_rate^(step / decay_steps).

    Use when: training from scratch and a smooth, gradual LR reduction is desired. A good
    default for most tabular and vision tasks.

    Trade-offs: Never reaches zero, so the optimizer always has a non-trivial step size.
    Sensitive to the choice of decay_rate and decay_steps — too aggressive and the model
    under-trains in the later stages; too gentle and the LR barely changes.
    """
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
    """Cosine annealing with warm restarts (Loshchilov & Hutter, 2017).

    The LR follows a cosine curve from base_lr down to eta_min over T_0 steps, then
    restarts. Each restart can have a longer period controlled by t_mult.

    Use when: you want periodic exploration boosts during training (each restart resets LR
    upward), which can help escape local minima. Popular for image classification.

    Trade-offs: The periodic restarts add variance to the loss curve. Selecting good T_0
    and T_mult values requires some tuning. Not well-suited for very short training runs.
    """
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


def init_one_cycle(
    config: LRSchedulerConfig,
    optimizer: Optimizer,
    step_info: StepInfo,
) -> OneCycleLR:
    """1cycle policy (Smith & Topin, 2018).

    Three phases: LR rises from initial_lr to max_lr over pct_start of total steps
    (warmup), then decays to min_lr = initial_lr / final_div_factor via cosine annealing.
    initial_lr = max_lr / div_factor.

    Use when: you want the fastest convergence in a single training run, especially for
    image/audio tasks. Works best with a pre-determined total_steps budget.

    Trade-offs: Requires total_steps to be known in advance (cannot restart mid-training
    without reinitializing). Not composable with an external warmup scheduler (warmup is
    built in). The LR goes to near-zero at the end, so extending training is ineffective.
    """
    total_steps = step_info.num_training_steps
    if not total_steps:
        total_steps = 1  # Avoid division-by-zero during dummy init

    # Determine max_lr: use config override or fall back to optimizer's param-group lr.
    if config.max_lr is not None:
        max_lr = config.max_lr
    else:
        max_lr = max(pg["lr"] for pg in optimizer.param_groups)

    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
        anneal_strategy="cos",
    )


def inverse_sqrt_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    """Inverse square root decay: lr = base_lr / sqrt(max(step, warmup_steps)).

    This is the original Transformer learning rate schedule from Vaswani et al. (2017).
    The LR warms up linearly for warmup_steps steps (handled externally via SequentialLR),
    then decays as 1/sqrt(step), which is slow and keeps the LR meaningfully large for a
    long time.

    The peak step count is anchored to inverse_sqrt_warmup_steps so that the decay
    transition is predictable regardless of the global warmup setting.

    Use when: training vanilla Transformer models from scratch (NLP, speech). Standard in
    sequence-to-sequence and language model pretraining baselines.

    Trade-offs: LR never reaches zero, so training can be extended freely. Decays slower
    than linear or exponential — may be too high for very long runs without tuning
    warmup_steps.
    """
    warmup = float(config.inverse_sqrt_warmup_steps)
    return 1.0 / math.sqrt(max(float(current_step), warmup))


def polynomial_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    """Polynomial decay with configurable power and end LR.

    lr = (base_lr - end_lr) * (1 - progress)^power + end_lr where progress = (step -
    warmup_steps) / (total_steps - warmup_steps).

    With power=1.0 this is equivalent to linear decay. Higher powers give more concave
    curves that stay high for longer and drop sharply near the end.

    Use when: replicating BERT, RoBERTa, or GPT-2 fine-tuning recipes. These models
    commonly use polynomial (linear) decay with a small warmup fraction.

    Trade-offs: Requires total_steps to be known. end_lr > 0 prevents the LR from
    collapsing to zero, which can help with very long training. Power > 1 risks a sudden
    sharp drop near the end of training.
    """
    if num_training_steps <= num_warmup_steps:
        return 1.0

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    progress = min(progress, 1.0)

    # Compute the scale factor as a fraction of the base LR (base LR = 1.0 in LambdaLR).
    end_lr_fraction = config.polynomial_end_lr  # relative to base_lr; 0.0 means decay to 0
    return (1.0 - end_lr_fraction) * math.pow(1.0 - progress, config.polynomial_power) + end_lr_fraction


def wsd_decay(current_step: int, num_training_steps: int, num_warmup_steps: int, config: LRSchedulerConfig):
    """Warmup-Stable-Decay (WSD) schedule.

    Three phases:

    1. Warmup: LR rises linearly from 0 to base_lr over wsd_warmup_fraction * total_steps
       steps.
    2. Stable: LR stays constant at base_lr for wsd_stable_fraction * total_steps steps.
    3. Decay: LR decreases via cosine from base_lr to 0 over wsd_decay_fraction *
       total_steps steps.

    wsd_warmup_fraction + wsd_stable_fraction + wsd_decay_fraction should sum to 1.

    Popular for LLM pretraining (MiniCPM, DeepSeek-V2). The long stable phase allows easy
    checkpoint reuse: you can extend training by appending another stable phase and a final
    decay.

    Use when: pretraining large language models where you want flexibility to extend the
    training budget without restarting from scratch.

    Trade-offs: The sharp cosine decay at the end can cause instability if decay_fraction
    is too small. The three fractions must be manually tuned and must sum to 1. Warmup is
    managed internally — do not also set warmup_fraction or warmup_evaluations.
    """
    T = float(num_training_steps)
    t = float(current_step)

    warmup_end = config.wsd_warmup_fraction * T
    stable_end = warmup_end + config.wsd_stable_fraction * T

    if t < warmup_end:
        # Linear warmup phase
        return t / max(1.0, warmup_end)
    elif t < stable_end:
        # Constant LR phase
        return 1.0
    else:
        # Cosine decay phase
        decay_start = stable_end
        decay_end = T
        decay_progress = (t - decay_start) / max(1.0, decay_end - decay_start)
        decay_progress = min(decay_progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


decay_registry = {
    None: wrap_decay_fn(no_decay),
    "linear": wrap_decay_fn(linear_decay),
    "exponential": wrap_decay_fn(exponential_decay),
    "cosine": init_cosine_decay,
    "inverse_sqrt": wrap_decay_fn(inverse_sqrt_decay),
    "polynomial": wrap_decay_fn(polynomial_decay),
    "wsd": wrap_decay_fn(wsd_decay),
    # one_cycle is handled specially in get_schedule_with_warmup_and_decay
}
