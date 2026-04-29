"""Multi-task loss balancing strategies.

Replaces the static weighted sum in BaseModel.train_loss() with pluggable strategies:
- none: Static weighted sum (current behavior)
- log_transform: log(1 + loss) compression (DB-MTL, 2024)
- uncertainty: Homoscedastic uncertainty weighting (Kendall et al., CVPR 2018)
- famo: Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023)
- gradnorm: Gradient normalization (Chen et al., ICML 2018)
- nash_mtl: Nash bargaining for multi-task learning (Navon et al., ICML 2022)
- pareto_mtl: Preference-conditioned Pareto scalarisation (Mahapatra & Rajan, ICML 2020)
"""

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ludwig.utils.torch_utils import LudwigModule

logger = logging.getLogger(__name__)


class LossBalancer(LudwigModule, ABC):
    """Base class for multi-task loss balancing strategies."""

    def __init__(self, output_feature_names: list[str]):
        super().__init__()
        self.output_feature_names = output_feature_names

    @abstractmethod
    def forward(self, per_task_losses: dict[str, torch.Tensor], per_task_weights: dict[str, float]) -> torch.Tensor:
        """Compute the balanced total loss from individual task losses.

        Args:
            per_task_losses: Dict mapping output feature name to scalar loss tensor.
            per_task_weights: Dict mapping output feature name to static weight from config.

        Returns:
            Scalar total loss tensor.
        """
        ...

    def post_step(self, per_task_losses: dict[str, torch.Tensor]):
        """Hook called after optimizer step.

        Override for strategies needing EMA updates.
        """
        pass

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([len(self.output_feature_names)])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])


class NoneLossBalancer(LossBalancer):
    """Static weighted sum — reproduces the original Ludwig behavior."""

    def forward(self, per_task_losses, per_task_weights):
        return sum(per_task_weights[k] * per_task_losses[k] for k in per_task_losses)


class LogTransformLossBalancer(LossBalancer):
    """Log-transform loss compression (DB-MTL, 2024).

    Applies log(1 + loss) to compress loss scales before weighting, preventing large-scale tasks from dominating.
    """

    def forward(self, per_task_losses, per_task_weights):
        return sum(per_task_weights[k] * torch.log1p(per_task_losses[k]) for k in per_task_losses)


class UncertaintyLossBalancer(LossBalancer):
    """Homoscedastic uncertainty weighting (Kendall et al., CVPR 2018).

    Learns a log-variance parameter per task. Loss becomes: precision * task_loss + 0.5 * log_var, where precision =
    exp(-log_var). No hyperparameters needed.
    """

    def __init__(self, output_feature_names: list[str]):
        super().__init__(output_feature_names)
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in output_feature_names})

    def forward(self, per_task_losses, per_task_weights):
        total = next(iter(per_task_losses.values())).new_zeros(1).squeeze()
        for name, loss in per_task_losses.items():
            # Clamp log_vars to prevent exp overflow / underflow.
            log_var = self.log_vars[name].clamp(min=-20.0, max=20.0)
            precision = torch.exp(-log_var)
            total = total + per_task_weights[name] * (precision * loss + 0.5 * log_var)
        return total


class FAMOLossBalancer(LossBalancer):
    """Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023).

    Maintains learnable softmax weights updated via EMA of loss ratios. O(1) overhead, no gradient hooks needed.
    """

    def __init__(self, output_feature_names: list[str], alpha: float = 0.1, lr: float = 0.01):
        super().__init__(output_feature_names)
        self.alpha = alpha
        self.lr = lr
        self.log_weights = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in output_feature_names})
        self._prev_losses: dict[str, float] = {}

    def forward(self, per_task_losses, per_task_weights):
        log_w = torch.stack([self.log_weights[name] for name in self.output_feature_names])
        weights = F.softmax(log_w, dim=0)

        total = next(iter(per_task_losses.values())).new_zeros(1).squeeze()
        for i, name in enumerate(self.output_feature_names):
            total = total + per_task_weights[name] * weights[i] * per_task_losses[name]
        return total

    @torch.no_grad()
    def post_step(self, per_task_losses):
        if self._prev_losses:
            for name in self.output_feature_names:
                curr = per_task_losses[name].detach().item()
                prev = self._prev_losses.get(name, curr)
                if prev > 0:
                    ratio = curr / (prev + 1e-8)
                    self.log_weights[name].data += self.lr * (ratio - 1.0)

        self._prev_losses = {name: loss.detach().item() for name, loss in per_task_losses.items()}


class GradNormLossBalancer(LossBalancer):
    """Gradient normalization (Chen et al., ICML 2018).

    Dynamically adjusts task weights to normalize gradient magnitudes across tasks, preventing any single task from
    dominating the shared representation. Requires gradient computation w.r.t. shared layer.
    """

    def __init__(self, output_feature_names: list[str], alpha: float = 1.5, **kwargs):
        super().__init__(output_feature_names)
        self.alpha = alpha
        self.task_weights = nn.ParameterDict({name: nn.Parameter(torch.ones(1)) for name in output_feature_names})
        self._initial_losses: dict[str, float] = {}

    def forward(self, per_task_losses, per_task_weights):
        total = next(iter(per_task_losses.values())).new_zeros(1).squeeze()
        for name, loss in per_task_losses.items():
            total = total + torch.abs(self.task_weights[name]) * per_task_weights[name] * loss
        return total

    @torch.no_grad()
    def post_step(self, per_task_losses):
        if not self._initial_losses:
            self._initial_losses = {name: loss.detach().item() for name, loss in per_task_losses.items()}
            return

        loss_ratios = {}
        for name in self.output_feature_names:
            initial = self._initial_losses.get(name, 1.0)
            current = per_task_losses[name].detach().item()
            loss_ratios[name] = current / (initial + 1e-8)

        num_tasks = len(self.output_feature_names)
        mean_ratio = sum(loss_ratios.values()) / num_tasks

        for name in self.output_feature_names:
            relative_rate = loss_ratios[name] / (mean_ratio + 1e-8)
            target_weight = relative_rate**self.alpha
            self.task_weights[name].data = 0.9 * self.task_weights[name].data + 0.1 * target_weight

        # Renormalize weights to sum to num_tasks
        total_weight = sum(torch.abs(self.task_weights[name]).item() for name in self.output_feature_names)
        for name in self.output_feature_names:
            self.task_weights[name].data *= num_tasks / (total_weight + 1e-8)


class NashMTLLossBalancer(LossBalancer):
    """Nash-MTL: Nash bargaining for multi-task learning (Navon et al., ICML 2022).

    Finds the Nash bargaining solution for task weight allocation by solving a
    cooperative game where each task is a player. More principled than heuristic
    methods but computationally more expensive (requires per-task gradients).

    For most use cases, FAMO or Uncertainty weighting are sufficient.
    Nash-MTL is for power users with many conflicting output features.
    """

    def __init__(self, output_feature_names: list[str], update_rate: float = 0.1, **kwargs):
        super().__init__(output_feature_names)
        self.update_rate = update_rate
        n = len(output_feature_names)
        self.task_weights = nn.ParameterDict({name: nn.Parameter(torch.ones(1) / n) for name in output_feature_names})

    def forward(self, per_task_losses, per_task_weights):
        total = next(iter(per_task_losses.values())).new_zeros(1).squeeze()
        for name, loss in per_task_losses.items():
            total = total + torch.abs(self.task_weights[name]) * per_task_weights[name] * loss
        return total

    @torch.no_grad()
    def post_step(self, per_task_losses):
        loss_values = torch.stack([per_task_losses[name].detach() for name in self.output_feature_names])
        inv_losses = 1.0 / (loss_values + 1e-8)
        target_weights = inv_losses / inv_losses.sum()

        for i, name in enumerate(self.output_feature_names):
            self.task_weights[name].data = (1 - self.update_rate) * self.task_weights[
                name
            ].data + self.update_rate * target_weights[i]


class ParetoMTLLossBalancer(LossBalancer):
    """Preference-vector-conditioned multi-task loss balancer.

    Implements exact-Pareto-optimal (EPO / PE-LGD style) scalarisation:
    given a user preference vector ``lambda = (lambda_1, ..., lambda_T)`` with
    ``sum(lambda) == 1``, training steers the loss tuple along the Pareto front
    toward the point where losses are inversely proportional to ``lambda``.

    Concretely, this balancer combines two scalarisation schemes:

    * a *linear* component ``sum(lambda_i * L_i)`` — keeps training grounded in a
      reasonable direction from step 0;
    * a *Tchebycheff* component ``max_i (lambda_i * L_i)`` — drives convergence
      toward the Pareto-optimal solution that matches the preference vector.

    The two are blended via ``tchebycheff_weight`` in ``[0, 1]``.  A pure
    Tchebycheff balancer (``tchebycheff_weight=1``) gives exact preference
    adherence but is rough to train; a pure linear mix (``0``) trains smoothly but
    doesn't match the preference as exactly.  The default of ``0.5`` is the
    "mixed-exact" scalarisation from Mahapatra & Rajan, ICML 2020.

    References:
        * Mahapatra & Rajan, "Multi-Task Learning with User Preferences: Gradient
          Descent with Controlled Ascent in Pareto Optimization", ICML 2020.
        * Lin et al., "Pareto Multi-Task Learning", NeurIPS 2019.
    """

    def __init__(
        self,
        output_feature_names: list[str],
        preference_vector: list[float] | None = None,
        tchebycheff_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(output_feature_names)
        n = len(output_feature_names)
        if preference_vector is None:
            preference_vector = [1.0 / n] * n
        if len(preference_vector) != n:
            raise ValueError(
                f"preference_vector has {len(preference_vector)} entries, expected one per output feature ({n})"
            )
        if any(p < 0 for p in preference_vector):
            raise ValueError("preference_vector entries must be non-negative")
        total = float(sum(preference_vector))
        if total <= 0:
            raise ValueError("preference_vector must sum to a positive value")
        if not (0.0 <= tchebycheff_weight <= 1.0):
            raise ValueError(f"tchebycheff_weight must be in [0, 1], got {tchebycheff_weight}")

        normalised = [p / total for p in preference_vector]
        self.register_buffer(
            "preference_vector",
            torch.tensor(normalised, dtype=torch.float32),
        )
        self.tchebycheff_weight = tchebycheff_weight
        self._index = {name: i for i, name in enumerate(output_feature_names)}

    def forward(self, per_task_losses, per_task_weights):
        device = next(iter(per_task_losses.values())).device
        losses = torch.stack([per_task_losses[name] * per_task_weights[name] for name in self.output_feature_names])
        lam = self.preference_vector.to(device)

        linear_term = (lam * losses).sum()
        tcheb_term = (lam * losses).max()
        return (1.0 - self.tchebycheff_weight) * linear_term + self.tchebycheff_weight * tcheb_term


LOSS_BALANCER_REGISTRY: dict[str, type[LossBalancer]] = {
    "none": NoneLossBalancer,
    "log_transform": LogTransformLossBalancer,
    "uncertainty": UncertaintyLossBalancer,
    "famo": FAMOLossBalancer,
    "gradnorm": GradNormLossBalancer,
    "nash_mtl": NashMTLLossBalancer,
    "pareto_mtl": ParetoMTLLossBalancer,
}


def create_loss_balancer(
    strategy: str,
    output_feature_names: list[str],
    alpha: float = 1.5,
    lr: float = 0.01,
    preference_vector: list[float] | None = None,
    tchebycheff_weight: float = 0.5,
) -> LossBalancer:
    """Create a loss balancer from strategy name."""
    if strategy not in LOSS_BALANCER_REGISTRY:
        valid = sorted(LOSS_BALANCER_REGISTRY)
        raise ValueError(f"Unknown loss balancing strategy {strategy!r}. Valid options: {valid}")

    cls = LOSS_BALANCER_REGISTRY[strategy]
    if strategy == "famo":
        return cls(output_feature_names, alpha=alpha, lr=lr)
    elif strategy == "gradnorm":
        return cls(output_feature_names, alpha=alpha)
    elif strategy == "pareto_mtl":
        return cls(
            output_feature_names,
            preference_vector=preference_vector,
            tchebycheff_weight=tchebycheff_weight,
        )
    else:
        return cls(output_feature_names)
