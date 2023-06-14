import logging
from typing import Dict, List, Optional, Tuple

import torch

from ludwig.distributed.base import DistributedStrategy
from ludwig.models.llm import LLM
from ludwig.modules.loss_modules import RewardLoss
from ludwig.schema.trainer import RewardModelTrainerConfig
from ludwig.trainers.registry import register_trainer
from ludwig.trainers.trainer import Trainer
from ludwig.utils.defaults import default_random_seed

logger = logging.getLogger(__name__)


@register_trainer("reward_model")
class RewardModelTrainer(Trainer):
    @staticmethod
    def get_schema_cls():
        return RewardModelTrainerConfig

    def __init__(
        self,
        config: RewardModelTrainerConfig,
        model: LLM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        report_tqdm_to_ray=False,
        random_seed: float = default_random_seed,
        distributed: Optional[DistributedStrategy] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            config,
            model,
            resume,
            skip_save_model,
            skip_save_progress,
            skip_save_log,
            callbacks,
            report_tqdm_to_ray,
            random_seed,
            distributed,
            device,
            **kwargs,
        )

        # Save the reward model loss function
        self.reward_loss_function = RewardLoss({})

        # Save the reward model dataset parameters
        self.id_column = config["output_features"][0]["proc_column"]
        self.transcript_column = config["input_features"][0]["proc_column"]

    def train_step(
        self, inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], should_step: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Performs a single training step of the RLHF reward model.

        Params:
            inputs: A dictionary of input data, from feature name to tensor.
            targets: A dictionary of target data, from feature name to tensor.
            should_step: Whether to perform a step of the optimizer after computing gradients.

        Returns:
            A tuple of the loss tensor and a dictionary of loss for every output feature.
        """
        if not all(
            self.use_amp is False,
            self.evaluate_training_set is True,
        ):
            raise ValueError("Invalid trainer arguments for RLHF reward model")

        # Validate inputs and targets
        input_names_expected = [self.transcript_column]
        input_names_actual = list(inputs.keys())
        if not input_names_actual == input_names_expected:
            raise ValueError(
                f"Invalid reward model training data inputs, expect inputs {input_names_expected}, "
                f"got inputs {input_names_actual}."
            )
        target_names_expected = [self.id_column]
        target_names_actual = list(targets.keys())
        if not target_names_actual == target_names_expected:
            raise ValueError(
                f"Invalid reward model training data targets, expect targets {target_names_expected}, "
                f"got targets {target_names_actual}."
            )

        # Run forward-propagation of the chosen and rejected inputs
        with self.distributed.prepare_model_update(self.dist_model, should_step=should_step):
            # Obtain model predictions and loss
            model_output_chosen = self.dist_model(inputs[self.transcript_column][0])
            model_output_rejected = self.dist_model(inputs[self.transcript_column][1])
            loss = self.reward_loss_function(model_output_chosen, model_output_rejected)
            loss = loss / self.gradient_accumulation_steps
            all_losses = loss

        # Begin the backward pass
        variables = self.dist_model.parameters()
        self.distributed.backward(loss, self.dist_model)

        if not should_step:
            # Short-circuit the parameter updates if we are still accumulating gradients
            return loss, all_losses

        # Wait for gradient aggregation to complete before clipping the gradients
        # When using AMP, we need to do this before unscaling.
        # See: https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py
        self.distributed.wait_optimizer_synced(self.optimizer)

        if self.distributed.allow_clip_gradients():
            # Clip gradients
            self.clip_grads(variables)

        # Apply gradient updates
        with self.distributed.prepare_optimizer_update(self.optimizer):
            # Because we already synchronized above, we skip doing so here
            self.distributed.step(self.optimizer)

        self.distributed.zero_grad(self.optimizer)

        return loss, all_losses
