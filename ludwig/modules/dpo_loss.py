"""Direct Preference Optimization (DPO) loss function.

Implements the DPO loss from Rafailov et al., "Direct Preference Optimization:
Your Language Model is Secretly a Reward Model", NeurIPS 2023.

DPO trains a language model to prefer "chosen" completions over "rejected" ones
by optimizing:
    L_DPO = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

where log_ratio = log(pi(y|x)) - log(pi_ref(y|x)) for policy pi and reference pi_ref.
"""

import torch
import torch.nn.functional as F


def compute_token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probabilities from logits and labels.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len] with -100 for tokens to ignore

    Returns:
        Per-example sum of log probabilities (only for non-ignored tokens).
        Shape: [batch]
    """
    # Shift for next-token prediction: logits predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_log_probs = log_probs.gather(dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    # Mask out ignored tokens (-100)
    mask = shift_labels != -100
    per_token_log_probs = per_token_log_probs * mask

    # Sum log probs per example
    return per_token_log_probs.sum(dim=-1)


def dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor | None = None,
    reference_rejected_log_probs: torch.Tensor | None = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss.

    When reference log probs are None, uses the implicit reference model
    (the model's own log probs before training, approximated as 0).

    Args:
        policy_chosen_logits: [batch, seq_len, vocab_size] from policy model on chosen text
        policy_rejected_logits: [batch, seq_len, vocab_size] from policy model on rejected text
        chosen_labels: [batch, seq_len] token IDs for chosen, -100 for masked
        rejected_labels: [batch, seq_len] token IDs for rejected, -100 for masked
        reference_chosen_log_probs: [batch] pre-computed reference model log probs (optional)
        reference_rejected_log_probs: [batch] pre-computed reference model log probs (optional)
        beta: Temperature parameter controlling deviation from reference (typical: 0.1-0.5)
        label_smoothing: Smoothing factor for label targets (0 = no smoothing)
        loss_type: "sigmoid" (standard DPO) or "ipo" (identity preference optimization)

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    policy_chosen_log_probs = compute_token_log_probs(policy_chosen_logits, chosen_labels)
    policy_rejected_log_probs = compute_token_log_probs(policy_rejected_logits, rejected_labels)

    # Log ratios: log(pi(y|x)) - log(pi_ref(y|x))
    if reference_chosen_log_probs is not None:
        chosen_log_ratios = policy_chosen_log_probs - reference_chosen_log_probs
    else:
        chosen_log_ratios = policy_chosen_log_probs

    if reference_rejected_log_probs is not None:
        rejected_log_ratios = policy_rejected_log_probs - reference_rejected_log_probs
    else:
        rejected_log_ratios = policy_rejected_log_probs

    # Preference margin
    logits_diff = beta * (chosen_log_ratios - rejected_log_ratios)

    if loss_type == "sigmoid":
        # Standard DPO: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        if label_smoothing > 0:
            # Soft labels: interpolate between 1 and 0.5
            losses = -F.logsigmoid(logits_diff) * (1 - label_smoothing) - F.logsigmoid(-logits_diff) * label_smoothing
        else:
            losses = -F.logsigmoid(logits_diff)
    elif loss_type == "ipo":
        # Identity Preference Optimization: (logits_diff - 1/(2*beta))^2
        losses = (logits_diff - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unknown DPO loss type: {loss_type}. Use 'sigmoid' or 'ipo'.")

    # Rewards for logging
    chosen_rewards = beta * chosen_log_ratios.detach()
    rejected_rewards = beta * rejected_log_ratios.detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
