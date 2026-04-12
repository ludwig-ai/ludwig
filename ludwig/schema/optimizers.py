import math
from abc import ABC
from dataclasses import field
from typing import ClassVar

import torch

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

try:
    from transformers.optimization import Adafactor as _TransformersAdafactor
except Exception:
    _TransformersAdafactor = None

try:
    from schedulefree import AdamWScheduleFree as _AdamWScheduleFree
except Exception:
    _AdamWScheduleFree = None

try:
    import soap as _soap_module

    _SOAPOptimizer = getattr(_soap_module, "SOAP", None)
except Exception:
    _SOAPOptimizer = None

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema.metadata import OPTIMIZER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.utils.registry import Registry

optimizer_registry = Registry()


@DeveloperAPI
def register_optimizer(name: str):
    def wrap(optimizer_config: BaseOptimizerConfig):
        optimizer_registry[name] = (optimizer_config.optimizer_class, optimizer_config)
        return optimizer_config

    return wrap


@DeveloperAPI
def get_optimizer_cls(name: str):
    """Get the optimizer schema class from the optimizer schema class registry."""
    return optimizer_registry[name][1]


@DeveloperAPI
class BaseOptimizerConfig(schema_utils.LudwigBaseConfig, ABC):
    """Base class for optimizers. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding `torch.optim.Optimizer` class are copied over: check each class to check which attributes are
    different from the torch-specified defaults.
    """

    optimizer_class: ClassVar[torch.optim.Optimizer | None] = None
    "Class variable pointing to the corresponding `torch.optim.Optimizer` class."

    type: str
    """Name corresponding to an optimizer `ludwig.modules.optimization_modules.optimizer_registry`.

    Technically mutable, but attempting to load a derived optimizer with `type` set to a mismatched value will result in
    a `ValidationError`.
    """

    @property
    def is_paged(self) -> bool:
        """Returns True if the optimizer is a Paged optimizer."""
        return False

    @property
    def is_8bit(self) -> bool:
        """Returns True if the optimizer is an 8-bit optimizer."""
        return False


@DeveloperAPI
@register_optimizer(name="sgd")
class SGDOptimizerConfig(BaseOptimizerConfig):
    """Parameters for stochastic gradient descent."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.SGD
    """Points to `torch.optim.SGD`."""

    type: str = schema_utils.ProtectedString("sgd")
    """Must be 'sgd' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry` (default:
       'sgd')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD :
    momentum: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Momentum factor.",
        parameter_metadata=OPTIMIZER_METADATA["momentum"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Weight decay ($L2$ penalty).",
        parameter_metadata=OPTIMIZER_METADATA["weight_decay"],
    )

    dampening: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Dampening for momentum.",
        parameter_metadata=OPTIMIZER_METADATA["dampening"],
    )

    nesterov: bool = schema_utils.Boolean(
        default=False,
        description="Enables Nesterov momentum.",
        parameter_metadata=OPTIMIZER_METADATA["nesterov"],
    )


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="sgd_8bit")
    class SGD8BitOptimizerConfig(SGDOptimizerConfig):
        """Parameters for stochastic gradient descent."""

        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.SGD8bit

        type: str = schema_utils.ProtectedString("sgd_8bit")

        block_wise: bool = schema_utils.Boolean(
            default=False,
            description="Whether to use block wise update.",
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        @property
        def is_8bit(self) -> bool:
            return True


@DeveloperAPI
@register_optimizer(name="lbfgs")
class LBFGSOptimizerConfig(BaseOptimizerConfig):
    """Parameters for stochastic gradient descent."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.LBFGS
    """Points to `torch.optim.LBFGS`."""

    type: str = schema_utils.ProtectedString("lbfgs")
    """Must be 'lbfgs' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry` (default:
       'lbfgs')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
    max_iter: int = schema_utils.Integer(
        default=20,
        description="Maximum number of iterations per optimization step.",
        parameter_metadata=OPTIMIZER_METADATA["max_iter"],
    )

    max_eval: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="Maximum number of function evaluations per optimization step. Default: `max_iter` * 1.25.",
        parameter_metadata=OPTIMIZER_METADATA["max_eval"],
    )

    tolerance_grad: float = schema_utils.NonNegativeFloat(
        default=1e-07,
        description="Termination tolerance on first order optimality.",
        parameter_metadata=OPTIMIZER_METADATA["tolerance_grad"],
    )

    tolerance_change: float = schema_utils.NonNegativeFloat(
        default=1e-09,
        description="Termination tolerance on function value/parameter changes.",
        parameter_metadata=OPTIMIZER_METADATA["tolerance_change"],
    )

    history_size: int = schema_utils.Integer(
        default=100, description="Update history size.", parameter_metadata=OPTIMIZER_METADATA["history_size"]
    )

    line_search_fn: str = schema_utils.StringOptions(
        ["strong_wolfe"],
        default=None,
        allow_none=True,
        description="Line search function to use.",
        parameter_metadata=OPTIMIZER_METADATA["line_search_fn"],
    )


@DeveloperAPI
@register_optimizer(name="adam")
class AdamOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adam optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adam
    """Points to `torch.optim.Adam`."""

    type: str = schema_utils.ProtectedString("adam")
    """Must be 'adam' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adam')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay (L2 penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    amsgrad: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use the AMSGrad variant of this algorithm from the paper 'On the Convergence of Adam "
        "and Beyond'.",
        parameter_metadata=OPTIMIZER_METADATA["amsgrad"],
    )


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="adam_8bit")
    class Adam8BitOptimizerConfig(AdamOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.Adam8bit

        type: str = schema_utils.ProtectedString("adam_8bit")

        block_wise: bool = schema_utils.Boolean(
            default=True,
            description="Whether to use block wise update.",
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        @property
        def is_8bit(self) -> bool:
            return True

    @DeveloperAPI
    @register_optimizer(name="paged_adam")
    class PagedAdamOptimizerConfig(Adam8BitOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.PagedAdam

        type: str = schema_utils.ProtectedString("paged_adam")

        @property
        def is_paged(self) -> bool:
            return True

        @property
        def is_8bit(self) -> bool:
            return False

    @DeveloperAPI
    @register_optimizer(name="paged_adam_8bit")
    class PagedAdam8BitOptimizerConfig(PagedAdamOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.PagedAdam8bit

        type: str = schema_utils.ProtectedString("paged_adam_8bit")

        @property
        def is_8bit(self) -> bool:
            return True


@DeveloperAPI
@register_optimizer(name="adamw")
class AdamWOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamw optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.AdamW
    """Points to `torch.optim.AdamW`."""

    type: str = schema_utils.ProtectedString("adamw")
    """Must be 'adamw' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamw')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    amsgrad: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use the AMSGrad variant of this algorithm from the paper 'On the Convergence of Adam "
        "and Beyond'. ",
        parameter_metadata=OPTIMIZER_METADATA["amsgrad"],
    )


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="adamw_8bit")
    class AdamW8BitOptimizerConfig(AdamWOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.AdamW8bit

        type: str = schema_utils.ProtectedString("adamw_8bit")

        block_wise: bool = schema_utils.Boolean(
            default=True,
            description="Whether to use block wise update.",
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        @property
        def is_8bit(self) -> bool:
            return True

    @DeveloperAPI
    @register_optimizer(name="paged_adamw")
    class PagedAdamWOptimizerConfig(AdamW8BitOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.PagedAdamW

        type: str = schema_utils.ProtectedString("paged_adamw")

        @property
        def is_paged(self) -> bool:
            return True

        @property
        def is_8bit(self) -> bool:
            return False

    @DeveloperAPI
    @register_optimizer(name="paged_adamw_8bit")
    class PagedAdamW8BitOptimizerConfig(PagedAdamWOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.PagedAdamW8bit

        type: str = schema_utils.ProtectedString("paged_adamw_8bit")

        @property
        def is_8bit(self) -> bool:
            return True


@DeveloperAPI
@register_optimizer(name="adadelta")
class AdadeltaOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adadelta optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adadelta
    """Points to `torch.optim.Adadelta`."""

    type: str = schema_utils.ProtectedString("adadelta")
    """Must be 'adadelta' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adadelta')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta :
    rho: float = schema_utils.FloatRange(
        default=0.9,
        min=0,
        max=1,
        description="Coefficient used for computing a running average of squared gradients.",
        parameter_metadata=OPTIMIZER_METADATA["rho"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-06,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )


@DeveloperAPI
@register_optimizer(name="adagrad")
class AdagradOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adagrad optimization."""

    # Example docstring
    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adagrad
    """Points to `torch.optim.Adagrad`."""

    type: str = schema_utils.ProtectedString("adagrad")
    """Must be 'adagrad' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adagrad')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad :
    initial_accumulator_value: float = schema_utils.NonNegativeFloat(
        default=0, description="", parameter_metadata=OPTIMIZER_METADATA["initial_accumulator_value"]
    )

    lr_decay: float = schema_utils.FloatRange(
        default=0, description="Learning rate decay.", parameter_metadata=OPTIMIZER_METADATA["lr_decay"]
    )

    weight_decay: float = schema_utils.FloatRange(
        default=0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    eps: float = schema_utils.FloatRange(
        default=1e-10,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="adagrad_8bit")
    class Adagrad8BitOptimizerConfig(AdagradOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.Adagrad8bit

        type: str = schema_utils.ProtectedString("adagrad_8bit")

        block_wise: bool = schema_utils.Boolean(
            default=True,
            description="Whether to use block wise update.",
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        @property
        def is_8bit(self) -> bool:
            return True


@DeveloperAPI
@register_optimizer(name="adamax")
class AdamaxOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamax optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adamax
    """Points to `torch.optim.Adamax`."""

    type: str = schema_utils.ProtectedString("adamax")
    """Must be 'adamax' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamax')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax :
    betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )


@DeveloperAPI
@register_optimizer(name="nadam")
class NadamOptimizerConfig(BaseOptimizerConfig):
    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.NAdam
    """Points to `torch.optim.NAdam`."""

    type: str = schema_utils.ProtectedString("nadam")

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam :

    betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    momentum_decay: float = schema_utils.NonNegativeFloat(
        default=4e-3, description="Momentum decay.", parameter_metadata=OPTIMIZER_METADATA["momentum_decay"]
    )


@DeveloperAPI
@register_optimizer(name="rmsprop")
class RMSPropOptimizerConfig(BaseOptimizerConfig):
    """Parameters for rmsprop optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.RMSprop
    """Points to `torch.optim.RMSprop`."""

    type: str = schema_utils.ProtectedString("rmsprop")
    """Must be 'rmsprop' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'rmsprop')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop:
    momentum: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Momentum factor.",
        parameter_metadata=OPTIMIZER_METADATA["momentum"],
    )

    alpha: float = schema_utils.NonNegativeFloat(
        default=0.99,
        description="Smoothing constant.",
        parameter_metadata=OPTIMIZER_METADATA["alpha"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    centered: bool = schema_utils.Boolean(
        default=False,
        description="If True, computes the centered RMSProp, and the gradient is normalized by an estimation of its "
        "variance.",
        parameter_metadata=OPTIMIZER_METADATA["centered"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="rmsprop_8bit")
    class RMSProp8BitOptimizerConfig(RMSPropOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.RMSprop8bit

        type: str = schema_utils.ProtectedString("rmsprop_8bit")

        block_wise: bool = schema_utils.Boolean(
            default=True,
            description="Whether to use block wise update.",
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        @property
        def is_8bit(self) -> bool:
            return True


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="lamb")
    class LAMBOptimizerConfig(BaseOptimizerConfig):
        """Layer-wise Adaptive Moments optimizer for Batch training.

        Paper: https://arxiv.org/pdf/1904.00962.pdf
        """

        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.LAMB

        type: str = schema_utils.ProtectedString("lamb")

        bias_correction: bool = schema_utils.Boolean(
            default=True,
        )

        betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
            default=(0.9, 0.999),
            description="Coefficients used for computing running averages of gradient and its square.",
            parameter_metadata=OPTIMIZER_METADATA["betas"],
        )

        eps: float = schema_utils.NonNegativeFloat(
            default=1e-08,
            description="Term added to the denominator to improve numerical stability.",
            parameter_metadata=OPTIMIZER_METADATA["eps"],
        )

        weight_decay: float = schema_utils.NonNegativeFloat(
            default=0.0,
            description="Weight decay (L2 penalty).",
            parameter_metadata=OPTIMIZER_METADATA["weight_decay"],
        )

        amsgrad: bool = schema_utils.Boolean(
            default=False,
            description=(
                "Whether to use the AMSGrad variant of this algorithm from the paper "
                "'On the Convergence of Adam and Beyond'."
            ),
            parameter_metadata=OPTIMIZER_METADATA["amsgrad"],
        )

        adam_w_mode: bool = schema_utils.Boolean(
            default=True,
            description="Whether to use the AdamW mode of this algorithm from the paper "
            "'Decoupled Weight Decay Regularization'.",
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        block_wise: bool = schema_utils.Boolean(
            default=False,
            description="Whether to use block wise update.",
        )

        max_unorm: float = schema_utils.FloatRange(
            default=1.0,
            min=0.0,
            max=1.0,
        )

    @DeveloperAPI
    @register_optimizer(name="lamb_8bit")
    class LAMB8BitOptimizerConfig(LAMBOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.LAMB8bit

        type: str = schema_utils.ProtectedString("lamb_8bit")

        @property
        def is_8bit(self) -> bool:
            return True


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="lars")
    class LARSOptimizerConfig(BaseOptimizerConfig):
        """Layerwise Adaptive Rate Scaling.

        Paper: https://arxiv.org/pdf/1708.03888.pdf
        """

        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.LARS

        type: str = schema_utils.ProtectedString("lars")

        # 0.9 taken from the original paper - momentum requires a non zero value
        # https://arxiv.org/pdf/1708.03888v3.pdf
        momentum: float = schema_utils.FloatRange(
            default=0.9,
            min=0.0,
            max=1.0,
            min_inclusive=False,
            description="Momentum factor.",
            parameter_metadata=OPTIMIZER_METADATA["momentum"],
        )

        dampening: float = schema_utils.FloatRange(
            default=0.0,
            min=0.0,
            max=1.0,
            description="Dampening for momentum.",
            parameter_metadata=OPTIMIZER_METADATA["dampening"],
        )

        weight_decay: float = schema_utils.NonNegativeFloat(
            default=0.0,
            description="Weight decay (L2 penalty).",
            parameter_metadata=OPTIMIZER_METADATA["weight_decay"],
        )

        nesterov: bool = schema_utils.Boolean(
            default=False,
            description="Enables Nesterov momentum.",
            parameter_metadata=OPTIMIZER_METADATA["nesterov"],
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        max_unorm: float = schema_utils.FloatRange(
            default=1.0,
            min=0.0,
            max=1.0,
        )

    @DeveloperAPI
    @register_optimizer(name="lars_8bit")
    class LARS8BitOptimizerConfig(LARSOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.LARS8bit

        type: str = schema_utils.ProtectedString("lars_8bit")

        @property
        def is_8bit(self) -> bool:
            return True


if bnb is not None:

    @DeveloperAPI
    @register_optimizer(name="lion")
    class LIONOptimizerConfig(BaseOptimizerConfig):
        """Evolved Sign Momentum.

        Paper: https://arxiv.org/pdf/2302.06675.pdf
        """

        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.Lion

        type: str = schema_utils.ProtectedString("lion")

        betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
            default=(0.9, 0.999),
            description="Coefficients used for computing running averages of gradient and its square.",
            parameter_metadata=OPTIMIZER_METADATA["betas"],
        )

        weight_decay: float = schema_utils.NonNegativeFloat(
            default=0.0,
            description="Weight decay (L2 penalty).",
            parameter_metadata=OPTIMIZER_METADATA["weight_decay"],
        )

        percentile_clipping: int = schema_utils.IntegerRange(
            default=100,
            min=0,
            max=100,
            description="Percentile clipping.",
        )

        block_wise: bool = schema_utils.Boolean(
            default=True,
            description="Whether to use block wise update.",
        )

    @DeveloperAPI
    @register_optimizer(name="lion_8bit")
    class LION8BitOptimizerConfig(LIONOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.Lion8bit

        type: str = schema_utils.ProtectedString("lion_8bit")

        @property
        def is_8bit(self) -> bool:
            return True

    @DeveloperAPI
    @register_optimizer(name="paged_lion")
    class PagedLionOptimizerConfig(LIONOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.PagedLion

        type: str = schema_utils.ProtectedString("paged_lion")

        @property
        def is_paged(self) -> bool:
            return True

    @DeveloperAPI
    @register_optimizer(name="paged_lion_8bit")
    class PagedLion8BitOptimizerConfig(PagedLionOptimizerConfig):
        optimizer_class: ClassVar[torch.optim.Optimizer] = bnb.optim.PagedLion8bit

        type: str = schema_utils.ProtectedString("paged_lion_8bit")

        @property
        def is_8bit(self) -> bool:
            return True


# ---------------------------------------------------------------------------
# Modern optimizers
# ---------------------------------------------------------------------------


@DeveloperAPI
@register_optimizer(name="radam")
class RAdamOptimizerConfig(BaseOptimizerConfig):
    """Rectified Adam (RAdam) optimizer config (Liu et al., 2020).

    Paper: https://arxiv.org/abs/1908.03265

    Convergence: Warms up the adaptive learning rate by computing an analytical
    approximation to the variance of the second moment estimate. This eliminates the
    need for an explicit warmup schedule: training is stable from step 1 even with a
    large learning rate. Converges to the same quality as Adam but with a wider range
    of valid learning rates.

    Memory footprint: Same as Adam -- two moment buffers per parameter (~2x model size
    in optimizer state).

    When to use: Any setting where Adam is unstable early in training without warmup.
    Especially useful for experimentation where you do not want to tune the warmup
    duration. Drop-in replacement for Adam/AdamW with no warmup scheduler needed.

    Common pitfalls: RAdam provides no benefit over well-warmed-up Adam in late training.
    The rectification term switches off after the variance stabilises (around step ~5-6
    at default beta2=0.999), so expect identical behaviour to Adam from that point on.
    """

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.RAdam
    """Points to `torch.optim.RAdam`."""

    type: str = schema_utils.ProtectedString("radam")
    """Must be 'radam' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`."""

    # Defaults from https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html
    betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Weight decay (L2 penalty).",
        parameter_metadata=OPTIMIZER_METADATA["weight_decay"],
    )


@DeveloperAPI
@register_optimizer(name="adafactor")
class AdafactorOptimizerConfig(BaseOptimizerConfig):
    """Adafactor optimizer config (Shazeer & Stern, 2018).

    Paper: https://arxiv.org/abs/1805.09843
    Implementation: `transformers.optimization.Adafactor`

    Convergence: Factorizes the second moment matrix into row and column factors instead
    of storing a full per-parameter tensor, dramatically reducing memory. Convergence is
    comparable to Adam on large Transformer models but can be slightly slower to converge
    on smaller tasks.

    Memory footprint: Very low -- O(n+m) per parameter matrix (row + column factors)
    instead of O(n*m). For a 1B parameter model this can save ~8 GB of optimizer state
    versus Adam, making it the go-to optimizer when GPU memory is the bottleneck.

    When to use: Training or fine-tuning very large language models (T5, LLaMA, GPT)
    where Adam's two-moment buffers exceed available GPU memory. Enabled by default in
    many Hugging Face T5 training recipes.

    Common pitfalls: When `relative_step=True` (default) Adafactor computes its own
    learning rate schedule -- do NOT combine with an external LR scheduler (set
    `lr=None`). When `relative_step=False` you must pass an explicit `lr`. Setting
    `scale_parameter=False` and `relative_step=False` with a manual `lr` is the
    standard recipe for fine-tuning.
    """

    optimizer_class: ClassVar[torch.optim.Optimizer | None] = _TransformersAdafactor
    """Points to `transformers.optimization.Adafactor` (None if transformers not installed)."""

    type: str = schema_utils.ProtectedString("adafactor")
    """Must be 'adafactor' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`."""

    # Adafactor manages its own LR schedule when relative_step=True, so lr defaults to None.
    lr: float | None = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        min=0.0,
        description=(
            "Learning rate. Set to None (default) when `relative_step=True` so that Adafactor manages "
            "its own schedule. Must be provided when `relative_step=False`."
        ),
    )

    scale_parameter: bool = schema_utils.Boolean(
        default=True,
        description=(
            "If True, the learning rate is scaled by the root mean square of the parameters. "
            "Should be True when `relative_step=True`."
        ),
    )

    relative_step: bool = schema_utils.Boolean(
        default=True,
        description=(
            "If True, a time-dependent learning rate is computed instead of using the external `lr`. "
            "Do not combine with an external LR scheduler."
        ),
    )

    warmup_init: bool = schema_utils.Boolean(
        default=False,
        description=(
            "If True, the time-dependent learning rate is linearly increased at initialization. "
            "Only effective when `relative_step=True`."
        ),
    )

    def __post_init__(self):
        if self.optimizer_class is None:
            raise ImportError(
                "The 'adafactor' optimizer requires the `transformers` package. "
                "Install it with: pip install transformers"
            )


@DeveloperAPI
@register_optimizer(name="schedule_free_adamw")
class ScheduleFreeAdamWOptimizerConfig(BaseOptimizerConfig):
    """Schedule-Free AdamW optimizer config (Defazio & Mishchenko, 2024).

    Paper: https://arxiv.org/abs/2405.15682
    Package: `schedulefree` (install with: pip install schedulefree)

    Convergence: Eliminates the need for a learning rate scheduler by maintaining a
    Polyak-Ruppert averaged iterate in addition to the standard momentum buffer. The
    averaged iterate is used for evaluation while the momentum buffer drives the
    optimization. Achieves performance comparable to or better than well-tuned
    cosine/linear decay schedules on a wide range of tasks.

    Memory footprint: Slightly higher than AdamW -- stores an extra averaged parameter
    buffer (z), so ~3x model size in optimizer state (vs 2x for AdamW).

    When to use: When you want to skip learning rate scheduler tuning entirely: no
    cosine decay, no linear warmup schedule (beyond the built-in `warmup_steps`). Ideal
    for rapid prototyping and hyperparameter sweeps where schedule tuning is expensive.
    Also useful for online/continual learning without a fixed horizon.

    Common pitfalls: Must call `optimizer.train()` before the training loop and
    `optimizer.eval()` before evaluation/inference -- the model is in a different state
    depending on which iterate (momentum vs averaged) is active. Forgetting these calls
    leads to degraded evaluation metrics. The `warmup_steps` parameter is built into
    the optimizer and replaces the external warmup scheduler.
    """

    optimizer_class: ClassVar[torch.optim.Optimizer | None] = _AdamWScheduleFree
    """Points to `schedulefree.AdamWScheduleFree` (None if schedulefree not installed)."""

    type: str = schema_utils.ProtectedString("schedule_free_adamw")
    """Must be 'schedule_free_adamw' - corresponds to name in optimizer_registry."""

    betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Weight decay (decoupled L2 penalty).",
        parameter_metadata=OPTIMIZER_METADATA["weight_decay"],
    )

    warmup_steps: int = schema_utils.Integer(
        default=0,
        description=(
            "Number of linear warmup steps built into the optimizer. "
            "Replaces an external warmup scheduler -- do not combine with one."
        ),
    )

    def __post_init__(self):
        if self.optimizer_class is None:
            raise ImportError(
                "The 'schedule_free_adamw' optimizer requires the `schedulefree` package. "
                "Install it with: pip install schedulefree"
            )


# ---------------------------------------------------------------------------
# Muon: pure-Python Newton-Schulz implementation so no extra package required
# ---------------------------------------------------------------------------


class _MuonOptimizer(torch.optim.Optimizer):
    """Muon -- Momentum + Orthogonalization via Newton-Schulz (Jordan et al., 2024).

    Paper: https://arxiv.org/abs/2409.20325
    """

    _NS_COEFFS = (3.4445, -4.7750, 2.0315)

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov}
        super().__init__(params, defaults)

    @torch.no_grad()
    def _zeropower_via_newtonschulz5(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Newton-Schulz iteration to approximate the orthogonal factor of G."""
        assert G.ndim >= 2
        a, b, c = self._NS_COEFFS
        X = G.bfloat16() if G.dtype not in (torch.float16, torch.bfloat16) else G
        X = X / (X.norm() + 1e-7)
        transposed = X.shape[-2] < X.shape[-1]
        if transposed:
            X = X.mT
        for _ in range(steps):
            A = X @ X.mT
            X = a * X + b * (A @ X) + c * (A @ A @ X)
        if transposed:
            X = X.mT
        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf

                if update.ndim >= 2:
                    orig_shape = update.shape
                    mat = update.view(orig_shape[0], -1)
                    mat = self._zeropower_via_newtonschulz5(mat)
                    scale = math.sqrt(max(mat.shape[-2], mat.shape[-1]))
                    update = mat.view(orig_shape) * scale

                p.add_(update, alpha=-lr)

        return loss


@DeveloperAPI
@register_optimizer(name="muon")
class MuonOptimizerConfig(BaseOptimizerConfig):
    """Muon optimizer config -- Momentum + Orthogonalization via Newton-Schulz (Jordan et al., 2024).

    Paper: https://arxiv.org/abs/2409.20325

    Convergence: Applies Nesterov SGD momentum then orthogonalizes each parameter update
    matrix via a fast Newton-Schulz iteration (5 steps). Keeps updates approximately
    orthogonal for more isotropic parameter changes. Empirically outperforms AdamW on
    large language model pre-training at equivalent compute.

    Memory footprint: Low -- stores only one momentum buffer per parameter (~1x model
    size), same as SGD with momentum. Significantly cheaper than Adam's two buffers.

    When to use: Pre-training large Transformer language models where you want faster
    convergence than AdamW at the same memory cost as SGD. Implemented in pure PyTorch
    (no extra package required).

    Common pitfalls: The NS iteration operates in bfloat16 by default for speed. The
    default lr (0.02) is much higher than typical Adam lr (1e-3); always re-tune lr
    when switching from Adam.
    """

    optimizer_class: ClassVar[torch.optim.Optimizer] = _MuonOptimizer
    """Points to the built-in `_MuonOptimizer` implementation."""

    type: str = schema_utils.ProtectedString("muon")
    """Must be 'muon' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`."""

    momentum: float = schema_utils.FloatRange(
        default=0.95,
        min=0.0,
        max=1.0,
        description="Momentum factor for Nesterov SGD applied before orthogonalization.",
        parameter_metadata=OPTIMIZER_METADATA["momentum"],
    )

    nesterov: bool = schema_utils.Boolean(
        default=True,
        description=(
            "If True, use Nesterov momentum (look-ahead gradient) before orthogonalization. "
            "The original Muon paper uses Nesterov."
        ),
        parameter_metadata=OPTIMIZER_METADATA["nesterov"],
    )


if _SOAPOptimizer is not None:

    @DeveloperAPI
    @register_optimizer(name="soap")
    class SOAPOptimizerConfig(BaseOptimizerConfig):
        """SOAP optimizer config -- Shampoo as Adam's Preconditioner (Vyas et al., 2024).

        Paper: https://arxiv.org/abs/2409.11321
        Package: `soap-pytorch` (install with: pip install soap-pytorch)

        Convergence: Maintains a Kronecker-factored (Shampoo-style) preconditioner for
        each weight matrix and runs Adam in its eigenbasis. Converges faster than AdamW
        in terms of iterations/tokens on large Transformer pre-training.

        Memory footprint: High -- stores Kronecker factors (m x m) and (n x n) per weight
        matrix (m, n) in addition to Adam's two moment buffers. 2-3x Adam's memory for
        typical Transformer shapes.

        When to use: Large-scale pre-training where compute is plentiful but wall-clock
        time is at a premium.

        Common pitfalls: The preconditioner update frequency trades off overhead vs
        freshness; a frequency of 10-100 steps is typical. Not recommended for small
        models where preconditioner overhead outweighs convergence gain.
        """

        optimizer_class: ClassVar[torch.optim.Optimizer] = _SOAPOptimizer
        """Points to `soap.SOAP` from the `soap-pytorch` package."""

        type: str = schema_utils.ProtectedString("soap")
        """Must be 'soap' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`."""

        betas: tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
            default=(0.95, 0.95),
            description=(
                "Coefficients for the first and second Adam moment estimates run in the "
                "Shampoo eigenbasis. Note: SOAP typically uses higher beta1 (0.95) than standard Adam."
            ),
            parameter_metadata=OPTIMIZER_METADATA["betas"],
        )

        weight_decay: float = schema_utils.NonNegativeFloat(
            default=0.01,
            description="Weight decay (decoupled L2 penalty, as in AdamW).",
            parameter_metadata=OPTIMIZER_METADATA["weight_decay"],
        )


@DeveloperAPI
def get_optimizer_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for optimizer in optimizer_registry:
        optimizer_cls = optimizer_registry[optimizer][1]
        other_props = schema_utils.unload_jsonschema_from_config_class(optimizer_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": optimizer},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


@DeveloperAPI
def OptimizerDataclassField(default="adam", description="", parameter_metadata: ParameterMetadata = None):
    """Custom dataclass field that when used inside of a dataclass will allow any optimizer in
    `ludwig.modules.optimization_modules.optimizer_registry`.

    Sets default optimizer to 'adam'.

    :param default: Dict specifying an optimizer with a `type` field and its associated parameters. Will attempt to use
           `type` to load optimizer from registry with given params. (default: {"type": "adam"}).
    :return: Initialized dataclass field that converts untyped dicts with params to optimizer dataclass instances.
    """

    class OptimizerSelection(schema_utils.TypeSelection):
        """Custom marshmallow field that deserializes a dict to a valid optimizer from
        `ludwig.modules.optimization_modules.optimizer_registry` and creates a corresponding `oneOf` JSON schema
        for external usage."""

        def __init__(self):
            super().__init__(
                registry=optimizer_registry,
                default_value=default,
                description=description,
                parameter_metadata=parameter_metadata,
            )

        def get_schema_from_registry(self, key: str) -> type[schema_utils.LudwigBaseConfig]:
            return get_optimizer_cls(key)

        def _jsonschema_type_mapping(self):
            # Note that this uses the same conditional pattern as combiners:
            return {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(optimizer_registry.keys()),
                        "default": default,
                        "description": "The type of optimizer to use during the learning process",
                    },
                },
                "title": "optimizer_options",
                "allOf": get_optimizer_conds(),
                "required": ["type"],
                "description": description,
            }

    return OptimizerSelection().get_default_field()


@DeveloperAPI
class GradientClippingConfig(schema_utils.LudwigBaseConfig):
    """Dataclass that holds gradient clipping parameters."""

    clipglobalnorm: float | None = schema_utils.FloatRange(
        default=0.5,
        allow_none=True,
        description="Maximum allowed norm of the gradients",
        parameter_metadata=OPTIMIZER_METADATA["gradient_clipping"],
    )

    # TODO(travis): is this redundant with `clipglobalnorm`?
    clipnorm: float | None = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        description="Maximum allowed norm of the gradients",
        parameter_metadata=OPTIMIZER_METADATA["gradient_clipping"],
    )

    clipvalue: float | None = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        description="Maximum allowed value of the gradients",
        parameter_metadata=OPTIMIZER_METADATA["gradient_clipping"],
    )


@DeveloperAPI
def GradientClippingDataclassField(description: str, default: dict = {}):
    """Returns custom dataclass field for `ludwig.modules.optimization_modules.GradientClippingConfig`. Allows
    `None` by default.

    :param description: Description of the gradient dataclass field
    :param default: dict that specifies clipping param values that will be loaded by its schema class (default: {}).
    """
    allow_none = True

    class GradientClippingConfigField(schema_utils.SchemaField):
        """Custom field class for gradient clipping.

        Deserializes a dict to a valid instance of `ludwig.modules.optimization_modules.GradientClippingConfig` and
        creates a corresponding JSON schema for external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value
            if isinstance(value, dict):
                try:
                    return GradientClippingConfig.model_validate(value)
                except (TypeError, ConfigValidationError):
                    raise ConfigValidationError(
                        f"Invalid params for gradient clipping: {value}, see GradientClippingConfig class."
                    )
            raise ConfigValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            return {
                "oneOf": [
                    {"type": "null", "title": "disabled", "description": "Disable gradient clipping."},
                    {
                        **schema_utils.unload_jsonschema_from_config_class(GradientClippingConfig),
                        "title": "enabled_options",
                    },
                ],
                "title": "gradient_clipping_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ConfigValidationError(f"Invalid default: `{default}`")

    def load_default():
        return GradientClippingConfig.model_validate(default)

    try:
        dump_default = GradientClippingConfig.model_validate(default).to_dict()
    except Exception:
        dump_default = default if isinstance(default, dict) else {}

    return field(
        metadata={
            "marshmallow_field": GradientClippingConfigField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(OPTIMIZER_METADATA["gradient_clipping"]),
                },
            )
        },
        default_factory=load_default,
    )
