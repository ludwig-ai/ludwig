import logging
from typing import Dict, List, Optional, Tuple

import torch

from ludwig.constants import MODEL_RWD
from ludwig.distributed.base import DistributedStrategy
from ludwig.models.ecd import RWD
from ludwig.modules.loss_modules import RewardLoss
from ludwig.schema.trainer import RWDTrainerConfig
from ludwig.trainers.registry import register_trainer
from ludwig.trainers.trainer import Trainer
from ludwig.utils.batch_size_tuner import BatchSizeEvaluator
from ludwig.utils.defaults import default_random_seed

logger = logging.getLogger(__name__)


@register_trainer(MODEL_RWD)
class RWDTrainer(Trainer):
    """This class trains models of type Reward Model."""

    @staticmethod
    def get_schema_cls():
        return RWDTrainerConfig

    def __init__(
        self,
        config: RWDTrainerConfig,
        model: RWD,
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

    def _create_batch_size_evaluator(self) -> BatchSizeEvaluator:
        trainer = self

        class _TrainerBatchSizeEvaluator(BatchSizeEvaluator):
            def reset(self):
                trainer.model.reset_metrics()
                trainer.optimizer.zero_grad()

            def step(self, batch_size: int):
                trainer.distributed.set_batch_size(trainer.dist_model, batch_size)
                inputs = {
                    input_feature_name: [
                        input_feature.create_sample_input(batch_size=batch_size).to(trainer.device),
                        input_feature.create_sample_input(batch_size=batch_size).to(trainer.device),
                    ]
                    for input_feature_name, input_feature in trainer.model.input_features.items()
                }
                targets = {
                    output_feature_name: output_feature.create_sample_output(batch_size=batch_size).to(trainer.device)
                    for output_feature_name, output_feature in trainer.model.output_features.items()
                }
                trainer.train_step(inputs, targets)

        return _TrainerBatchSizeEvaluator()

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
        if self.use_amp is True:
            raise ValueError("Invalid trainer arguments for RLHF reward model")

        # Validate inputs and targets
        if not len(inputs) == 1:
            raise ValueError(f"Invalid reward model training data inputs, expect 1 input feature, got {len(inputs)}.")
        if not len(targets) == 1:
            raise ValueError(
                f"Invalid reward model training data targets, expect 1 target feature, got {len(targets)}."
            )
        id_column = list(targets.keys())[0]
        transcript_column = list(inputs.keys())[0]

        # Run forward-propagation of the chosen and rejected inputs
        with self.distributed.prepare_model_update(self.dist_model, should_step=should_step):
            # Obtain model predictions and loss
            inputs_chosen = {transcript_column: inputs[transcript_column][0]}
            inputs_rejected = {transcript_column: inputs[transcript_column][1]}
            model_output_chosen = self.dist_model(inputs_chosen)
            model_output_rejected = self.dist_model(inputs_rejected)
            logits_chosen = model_output_chosen[f"{id_column}::logits"]
            logits_rejected = model_output_rejected[f"{id_column}::logits"]
            loss = self.reward_loss_function(logits_chosen, logits_rejected)
            loss = loss / self.gradient_accumulation_steps
            all_losses = {"reward_loss": loss}

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
