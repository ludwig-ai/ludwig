from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Mapping, TYPE_CHECKING

import deepspeed
import deepspeed.comm
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from packaging import version
from torch import nn
from torch.optim.optimizer import Optimizer

from ludwig.constants import MIN_POSSIBLE_BATCH_SIZE
from ludwig.distributed.ddp import DDPStrategy
from ludwig.modules.optimization_modules import get_optimizer_class_and_kwargs
from ludwig.utils.checkpoint_utils import Checkpoint, MultiNodeCheckpoint
from ludwig.utils.model_utils import extract_tensors, replace_tensors

_deepspeed_0101 = version.parse(deepspeed.__version__) >= version.parse("0.10.1")


if TYPE_CHECKING:
    from ludwig.modules.lr_scheduler import LRScheduler
    from ludwig.schema.trainer import ECDTrainerConfig

# By defaulting to deepspeed zero stage 3, we assume the worst-case scenario where the model may not fit into
# memory, necessitating model parallel + data parallel training due to its size. This choice also assumes no
# quantization-based fine-tuning. The goal is to provide a simple and robust configuration that "just works" for
# various LLM models, irrespective of their size. This configuration ensures the highest chance of success for
# full fine-tuning with or without adapters, even when specific model size reduction techniques are not specified
# in the LLM fine-tuning configuration.
DEFAULT_ZERO_OPTIMIZATION = {
    "stage": 3,
    "stage3_gather_16bit_weights_on_model_save": "auto",
    "offload_optimizer": {"device": "auto"},
    "offload_param": {"device": "auto"},
}

# Filter out warnings about DeepSpeed use of deprecated methods. Can remove on upgrade to DeepSpeed 0.9.
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module="torch.distributed.distributed_c10d",
)


class DeepSpeedStrategy(DDPStrategy):
    def __init__(
        self,
        zero_optimization: dict[str, Any] = None,
        fp16: dict[str, Any] = None,
        bf16: dict[str, Any] = None,
        compression_training: dict[str, Any] = None,
        **kwargs,
    ):
        # If we're initializing from a `deepspeed` CLI command, deepspeed will have already been initialized, as
        # indicated by the presence of the LOCAL_RANK var. Otherwise, we're initializing from Ray / torchrun, and will
        # need to set this var ourselves, then init DeepSpeed here.
        local_rank, local_size = os.environ.get("LOCAL_RANK"), os.environ.get("LOCAL_SIZE")
        init_deepspeed = local_rank is None or local_size is None

        super().__init__(**kwargs)

        # By defaulting to deepspeed zero stage 3, we assume the worst-case scenario where the model may not fit
        # into memory, necessitating model parallel + data parallel training due to its size. This choice also
        # assumes no quantization-based fine-tuning. The goal is to provide a simple and robust configuration
        # that "just works" for various LLM models, irrespective of their size. This configuration ensures the
        # highest chance of success for full fine-tuning with or without adapters, even when specific model
        # size reduction techniques are not specified in the LLM fine-tuning configuration.
        self.zero_optimization = zero_optimization or DEFAULT_ZERO_OPTIMIZATION
        self.zero_optimization_stage = self.zero_optimization.get("stage", 3)

        self.fp16 = fp16
        self.bf16 = bf16
        self.compression_training = compression_training

        if init_deepspeed:
            os.environ["LOCAL_RANK"] = str(self.local_rank())
            os.environ["LOCAL_SIZE"] = str(self.local_size())
            os.environ["RANK"] = str(self.rank())
            os.environ["WORLD_SIZE"] = str(self.size())
            deepspeed.init_distributed()

    def _log_on_init(self):
        logging.info("Using DeepSpeed strategy")

    def prepare(
        self,
        model: nn.Module,
        trainer_config: ECDTrainerConfig,
        base_learning_rate: float,
    ) -> tuple[nn.Module, Optimizer]:
        # If `batch_size=auto`, we set to MIN_POSSIBLE_BATCH_SIZE temporarily until auto-tuning adjusts it`
        # We can really set it to be whatever we want, as it will be overridden by the auto-tuning.
        batch_size = (
            trainer_config.batch_size if isinstance(trainer_config.batch_size, int) else MIN_POSSIBLE_BATCH_SIZE
        )

        # Paged and 8-bit optimizers are not supported by Deepspeed - just whatever is supported
        # by torch.optim.Optimizer. https://www.deepspeed.ai/docs/config-json/#optimizer-parameters.
        if trainer_config.optimizer.is_paged or trainer_config.optimizer.is_8bit:
            raise ValueError("Cannot use a paged or 8-bit optimizer with DeepSpeed.")

        optimizer_cls, optimizer_kwargs = get_optimizer_class_and_kwargs(trainer_config.optimizer, base_learning_rate)
        ds_config = {
            "amp": {
                "enabled": trainer_config.use_mixed_precision,
            },
            "optimizer": {
                "type": optimizer_cls.__name__,
                "params": optimizer_kwargs,
            },
            "zero_optimization": self.zero_optimization,
            "gradient_clipping": trainer_config.gradient_clipping.clipglobalnorm,
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": trainer_config.gradient_accumulation_steps,
            "steps_per_print": trainer_config.steps_per_checkpoint or 10000,
        }

        # DeepSpeed doesn't like passing these params as None values
        if self.fp16 is not None:
            ds_config["fp16"] = self.fp16
        if self.bf16 is not None:
            ds_config["bf16"] = self.bf16
        if self.compression_training is not None:
            ds_config["compression_training"] = self.compression_training

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            lr_scheduler=None,  # Don't let DeepSpeed manage the learning rate scheduler
            config=ds_config,
            dist_init_required=False,
        )

        if hasattr(optimizer, "optimizer"):
            # Zero-3 wraps the optimizer
            optimizer = optimizer.optimizer

        return model_engine, optimizer

    def prepare_for_inference(self, model: nn.Module) -> nn.Module:
        """Prepares the model for inference/evaluation.

        For DeepSpeed Stage 3, we need to wrap the model in a DeepSpeed engine for inference. For all other DeepSpeed
        stages, we load the base model back. For LLMs, this recreates either the base model or the PEFT model, depending
        on whether a PEFT adapter was specified.
        """
        if self.zero_optimization_stage != 3:
            # For all DeepSpeed stages that are not stage 3, we can just return the model.
            # The model doesn't require model parallelism, and can be placed on the GPUs directly.
            model.prepare_for_inference()
            return model

        # Only Zero3 models need to be wrapped in a DeepSpeed engine for inference.
        # DeepSpeed ZeRO Inference supports ZeRO stage 3 with ZeRO-Infinity. It uses the same ZeRO protocol as
        # training, but it doesn’t use an optimizer and a lr scheduler and only stage 3 is relevant.
        # Docs: https://huggingface.co/docs/transformers/main_classes/deepspeed#zero-inference
        ds_config = {}
        model_engine = deepspeed.init_inference(model=model, config=ds_config)
        return model_engine

    def to_device(self, model, device: torch.device = None) -> nn.Module:
        return model

    def backward(self, loss: torch.Tensor, model: nn.Module):
        # See: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py
        # runs backpropagation and handles mixed precision
        model.backward(loss)

        # Deepspeed's `engine.step` performs the following operations:
        # - gradient accumulation check
        # - gradient clipping
        # - optimizer step
        # - zero grad
        # - checking overflow
        # - lr_scheduler step (only if engine.lr_scheduler is not None)
        model.step()
        # and this plugin overrides the above calls with no-ops when Accelerate runs under
        # Deepspeed, but allows normal functionality for non-Deepspeed cases thus enabling a simple
        # training loop that works transparently under many training regimes.

    def step(self, optimizer: Optimizer, *args, **kwargs):
        # Handled by `self.backward(loss)`
        pass

    def zero_grad(self, optimizer: Optimizer):
        # Handled by `self.backward(loss)`
        pass

    def set_batch_size(self, model: nn.Module, batch_size: int):
        # Adapted from:
        # https://github.com/microsoft/DeepSpeed/blob/7ce371b139521b1ebbf052f0496b1a16397c1d19/deepspeed/runtime/engine.py#L422  # noqa: E501
        model._config.micro_batch_size_per_gpu = batch_size
        model._config.train_batch_size = batch_size * self.size() * model._config.gradient_accumulation_steps

    def barrier(self):
        deepspeed.comm.barrier()

    def allow_gradient_accumulation(self) -> bool:
        """DeepSpeed handles gradient accumulation internally."""
        return False

    def allow_mixed_precision(self) -> bool:
        """DeepSpeed handles mixed precision internally."""
        return False

    def allow_clip_gradients(self) -> bool:
        """DeepSpeed handles gradient clipping internally."""
        return False

    def prepare_before_load(self) -> bool:
        """DeepSpeed requires the engine to be re-initialized before loading.

        https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#loading-training-checkpoints
        """
        return True

    @classmethod
    def is_model_parallel(cls) -> bool:
        return True

    def create_checkpoint_handle(
        self,
        dist_model: nn.Module,
        model: nn.Module,
        optimizer: Optimizer = None,
        scheduler: LRScheduler = None,
    ) -> Checkpoint:
        """Creates a checkpoint handle for saving and loading checkpoints."""
        if self.zero_optimization_stage != 3:
            # For all DeepSpeed stages that are not stage 3, we should use the standard
            # PyTorch checkpointing mechanism because we do not need to shard the checkpoints as a
            # result of data parallel training.
            return MultiNodeCheckpoint(self, dist_model, optimizer, scheduler)

        # DeepSpeed Zero3 checkpoints are not compatible with the standard PyTorch checkpointing mechanism since
        # they are sharded across multiple files. We need to use DeepSpeed's checkpointing mechanism instead.
        return DeepSpeedCheckpoint(self, dist_model, optimizer, scheduler)

    @classmethod
    def extract_adapter_weights_for_serialization(cls, model: nn.Module) -> Mapping[str, Any]:
        """Grabs the adapter weight tensors from the model for serialization."""
        from peft.utils.save_and_load import get_peft_model_state_dict

        # We do model.model because the model is wrapped in a the LLM model wrapper and this function
        # expects a PeftModel object.
        return get_peft_model_state_dict(model.model)

    @classmethod
    def extract_model_for_serialization(
        cls, model: nn.Module, optimization_stage: int | None = None
    ) -> nn.Module | tuple[nn.Module, list[dict]]:
        """Extracts the models weights from the model as numpy tensors for serialization into the Ray object store.

        Model weights are only serialized for DeepSpeed Zero3 models. For all other DeepSpeed stages, we can just return
        the model.
        """
        if optimization_stage != 3:
            return model

        return extract_tensors(model)

    @classmethod
    def replace_adapter_weights_from_serialization(cls, model: nn.Module, state_dict: dict[str, Any]) -> nn.Module:
        """Replaces the adapter weight tensors in the model from the provided state dictionary."""
        from peft.utils.save_and_load import set_peft_model_state_dict

        # We do model.model because the model is wrapped in a the LLM model wrapper and this function
        # expects a PeftModel object.
        set_peft_model_state_dict(model.model, state_dict)
        return model

    @classmethod
    def replace_model_from_serialization(
        cls,
        state: nn.Module | tuple[nn.Module, list[dict]],
        optimization_stage: int | None = None,
    ) -> nn.Module | tuple[nn.Module, list[dict]]:
        """Replaces the serialized model weights from the provided state object.

        Model weights are only serialized for DeepSpeed Zero3 models. For all other DeepSpeed stages, we can just return
        the state.
        """
        if optimization_stage != 3:
            # For DeepSpeed stages 0, 1 and 2, we can just return the state.
            assert isinstance(state, nn.Module)
            return state

        assert isinstance(state, tuple)
        model, model_weights = state
        replace_tensors(model, model_weights, torch.device("cpu"))
        return model

    @property
    def optimization_stage(self) -> int | None:
        return self.zero_optimization_stage


class DeepSpeedCheckpoint(Checkpoint):
    def prepare(self, directory: str):
        if self.distributed.local_rank() == 0:
            # Checkpoints need to be written on every rank, but the directory only needs to be created once per node.
            super().prepare(directory)

    def load(self, save_path: str, device: torch.device = None) -> bool:
        """Load a checkpoint.

        For DeepSpeed, we need every worker to independently load back the model weights, as the checkpoints themselves
        may be sharded (when using DeepSpeed Zero3).

        https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#loading-training-checkpoints
        """
        _, client_state = self.model.load_checkpoint(save_path, load_lr_scheduler_states=False)
        self.global_step = self._get_global_step(client_state, save_path)
        if self.scheduler is not None and "scheduler_state" in client_state:
            self.scheduler.load_state_dict(client_state["scheduler_state"])
        return True

    def save(self, save_path: str, global_step: int):
        client_state = {
            "global_step": global_step,
        }
        if self.scheduler is not None:
            client_state["scheduler_state"] = self.scheduler.state_dict()

        kwargs = {}
        if _deepspeed_0101:
            kwargs["exclude_frozen_parameters"] = True

        self.model.save_checkpoint(save_path, client_state=client_state, **kwargs)

    def get_state_for_inference(self, save_path: str, device: torch.device = None) -> Mapping[str, Any]:
        if self.model.zero_optimization_stage() == 3:
            return get_fp32_state_dict_from_zero_checkpoint(save_path)

        self.model.load_checkpoint(
            save_path, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True
        )
        return self.model.module.cpu().state_dict()
