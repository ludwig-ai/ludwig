"""DPO (Direct Preference Optimization) trainer for LLM fine-tuning.

Trains a language model to prefer "chosen" completions over "rejected" ones
without needing a separate reward model. Based on Rafailov et al., NeurIPS 2023.

Expects data with columns: prompt, chosen (preferred completion), rejected (dispreferred completion).

Config:
    model_type: llm
    trainer:
      type: dpo
      beta: 0.1
      loss_type: sigmoid  # or ipo
"""

import logging

import torch

from ludwig.constants import LOGITS, USED_TOKENS
from ludwig.modules.dpo_loss import dpo_loss
from ludwig.trainers.registry import register_llm_trainer
from ludwig.trainers.trainer import Trainer

logger = logging.getLogger(__name__)


@register_llm_trainer("dpo")
class DPOTrainer(Trainer):
    """Direct Preference Optimization trainer.

    Requires input data with 'chosen' and 'rejected' text columns. The model learns to assign higher probability to
    chosen completions over rejected ones.

    The DPO loss replaces the standard next-token cross-entropy loss with a preference-based objective that implicitly
    learns a reward function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = getattr(self.config, "dpo_beta", 0.1)
        self.loss_type = getattr(self.config, "dpo_loss_type", "sigmoid")
        self.label_smoothing = getattr(self.config, "dpo_label_smoothing", 0.0)

        # Reference model log probs can be pre-computed and cached.
        # For simplicity, we use the implicit reference (no separate model).
        # To use an explicit reference model, set reference_free=False and
        # provide reference log probs during training.
        self._reference_chosen_log_probs = None
        self._reference_rejected_log_probs = None

        logger.info(f"DPO trainer initialized: beta={self.beta}, loss_type={self.loss_type}")

    def train_step(self, inputs, targets, should_step=True, profiler=None):
        """Override train_step to compute DPO loss instead of standard CE loss.

        Expects inputs to contain 'chosen' and 'rejected' keys with tokenized sequences. The model computes forward
        passes on both and uses DPO loss to train.
        """
        import contextlib

        with torch.amp.autocast("cuda") if self.use_amp else contextlib.nullcontext():
            with self.distributed.prepare_model_update(self.dist_model, should_step=should_step):
                # Forward pass on chosen completions
                chosen_outputs = self.dist_model((inputs, targets))
                chosen_logits = None
                for key, val in chosen_outputs.items():
                    if LOGITS in key:
                        chosen_logits = val
                        break

                # For DPO, we need a second forward pass on rejected completions.
                # In Ludwig's current data pipeline, chosen and rejected are separate
                # features. If 'rejected' targets exist, use them; otherwise fall back
                # to standard loss.
                rejected_targets = {}
                has_rejected = False
                for key in targets:
                    if "rejected" in key.lower():
                        rejected_targets[key.replace("rejected", "chosen").replace("_rejected", "")] = targets[key]
                        has_rejected = True

                if has_rejected and chosen_logits is not None:
                    # Forward pass on rejected completions
                    rejected_outputs = self.dist_model((inputs, rejected_targets))
                    rejected_logits = None
                    for key, val in rejected_outputs.items():
                        if LOGITS in key:
                            rejected_logits = val
                            break

                    if rejected_logits is not None:
                        # Get labels (token IDs) for chosen and rejected
                        chosen_labels = next(iter(targets.values()))
                        rejected_labels = next(iter(rejected_targets.values()))

                        loss, chosen_rewards, rejected_rewards = dpo_loss(
                            policy_chosen_logits=chosen_logits,
                            policy_rejected_logits=rejected_logits,
                            chosen_labels=chosen_labels,
                            rejected_labels=rejected_labels,
                            reference_chosen_log_probs=self._reference_chosen_log_probs,
                            reference_rejected_log_probs=self._reference_rejected_log_probs,
                            beta=self.beta,
                            label_smoothing=self.label_smoothing,
                            loss_type=self.loss_type,
                        )
                        loss = loss / self.gradient_accumulation_steps
                    else:
                        # Fallback to standard loss if rejected logits not available
                        loss, all_losses = self.model.train_loss(
                            targets, chosen_outputs, self.regularization_type, self.regularization_lambda
                        )
                        loss = loss / self.gradient_accumulation_steps
                else:
                    # No rejected data available, use standard CE loss
                    loss, all_losses = self.model.train_loss(
                        targets, chosen_outputs, self.regularization_type, self.regularization_lambda
                    )
                    loss = loss / self.gradient_accumulation_steps

        used_tokens = chosen_outputs.get(USED_TOKENS, torch.tensor(0))

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            self.distributed.backward(loss, self.dist_model)

        if not should_step:
            return loss, {}, used_tokens

        self.distributed.wait_optimizer_synced(self.optimizer)

        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        if self.distributed.allow_clip_gradients():
            self.clip_grads(self.dist_model.parameters())

        with self.distributed.prepare_optimizer_update(self.optimizer):
            if self.use_amp:
                self.scaler.step(self.optimizer)
            else:
                self.distributed.step(self.optimizer)

        if self.use_amp:
            self.scaler.update()

        self.distributed.zero_grad(self.optimizer)

        if profiler:
            profiler.step()

        return loss, {}, used_tokens
