"""Preference optimization loss functions beyond DPO.

Implements:
- KTO: Kahneman-Tversky Optimization (Ethayarajh et al., 2024)
- ORPO: Odds Ratio Preference Optimization (Hong et al., 2024)
- GRPO: Group Relative Policy Optimization (Shao et al., 2024, used in DeepSeek-R1)

All functions expect per-token logits and labels, and compute preference-based
losses that train models to prefer certain completions over others.
"""

import torch
import torch.nn.functional as F

from ludwig.modules.dpo_loss import compute_token_log_probs


def kto_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor | None = None,
    reference_rejected_log_probs: torch.Tensor | None = None,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kahneman-Tversky Optimization loss (Ethayarajh et al., 2024).

    Unlike DPO which requires paired chosen/rejected data, KTO can work with
    unpaired preferences (just "this is good" or "this is bad" labels). However,
    when both are available, it uses them together.

    The loss applies prospect theory: losses loom larger than gains, so the model
    is penalized more for generating rejected text than rewarded for chosen text.

    L_KTO = (1 - sigmoid(beta * (log_ratio_chosen - KL))) for chosen
          + (1 - sigmoid(beta * (KL - log_ratio_rejected))) for rejected

    where KL is the average KL divergence between policy and reference.
    """
    policy_chosen_lp = compute_token_log_probs(policy_chosen_logits, chosen_labels)
    policy_rejected_lp = compute_token_log_probs(policy_rejected_logits, rejected_labels)

    if reference_chosen_log_probs is not None:
        chosen_log_ratios = policy_chosen_lp - reference_chosen_log_probs
    else:
        chosen_log_ratios = policy_chosen_lp

    if reference_rejected_log_probs is not None:
        rejected_log_ratios = policy_rejected_lp - reference_rejected_log_probs
    else:
        rejected_log_ratios = policy_rejected_lp

    # KL divergence estimate (mean of absolute log ratios)
    kl = 0.5 * (chosen_log_ratios.abs().mean() + rejected_log_ratios.abs().mean())

    # KTO loss: asymmetric treatment of chosen vs rejected
    chosen_loss = 1 - F.sigmoid(beta * (chosen_log_ratios - kl))
    rejected_loss = 1 - F.sigmoid(beta * (kl - rejected_log_ratios))

    loss = 0.5 * (chosen_loss.mean() + rejected_loss.mean())

    chosen_rewards = beta * chosen_log_ratios.detach()
    rejected_rewards = beta * rejected_log_ratios.detach()

    return loss, chosen_rewards.mean(), rejected_rewards.mean()


def orpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Odds Ratio Preference Optimization loss (Hong et al., 2024).

    ORPO combines SFT and preference alignment into a single objective by
    using the odds ratio of chosen vs rejected log probabilities. Does not
    require a reference model at all.

    L_ORPO = L_SFT(chosen) - beta * log(odds_ratio)

    where odds_ratio = odds(chosen) / odds(rejected)
    and odds(y) = P(y) / (1 - P(y))
    """
    chosen_lp = compute_token_log_probs(policy_chosen_logits, chosen_labels)
    rejected_lp = compute_token_log_probs(policy_rejected_logits, rejected_labels)

    # Log odds ratio
    chosen_log_odds = chosen_lp - torch.log1p(-torch.exp(chosen_lp).clamp(max=1 - 1e-7))
    rejected_log_odds = rejected_lp - torch.log1p(-torch.exp(rejected_lp).clamp(max=1 - 1e-7))
    log_odds_ratio = chosen_log_odds - rejected_log_odds

    # SFT loss on chosen (standard next-token cross-entropy)
    shift_logits = policy_chosen_logits[:, :-1, :].contiguous()
    shift_labels = chosen_labels[:, 1:].contiguous()
    sft_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    # Combined loss
    loss = sft_loss - beta * F.logsigmoid(log_odds_ratio).mean()

    chosen_rewards = chosen_lp.detach()
    rejected_rewards = rejected_lp.detach()

    return loss, chosen_rewards.mean(), rejected_rewards.mean()


def grpo_loss(
    policy_logits: torch.Tensor,
    labels: torch.Tensor,
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor | None = None,
    reference_log_probs: torch.Tensor | None = None,
    beta: float = 0.04,
    epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group Relative Policy Optimization loss (Shao et al., 2024).

    GRPO is used in DeepSeek-R1 and DeepSeek-Math. Unlike PPO, it does not
    require a critic/value model. Instead, it uses group-level rewards:
    for each prompt, generate multiple completions, score them with a reward
    function, normalize rewards within the group, and use the normalized
    rewards as advantages.

    This function computes the GRPO objective for a single group of completions
    from the same prompt. The caller is responsible for generating multiple
    completions and computing rewards.

    L_GRPO = -mean(min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage))
             + beta * KL(policy || reference)

    where ratio = exp(log_prob_new - log_prob_old)
    and advantage = (reward - mean(reward)) / std(reward)

    Args:
        policy_logits: [group_size, seq_len, vocab_size] current policy logits
        labels: [group_size, seq_len] token IDs with -100 for masked positions
        rewards: [group_size] scalar rewards for each completion in the group
        old_log_probs: [group_size] log probs from the old policy (for importance sampling)
        reference_log_probs: [group_size] log probs from the reference model (for KL penalty)
        beta: KL penalty coefficient
        epsilon: PPO-style clipping parameter

    Returns:
        (loss, mean_advantage)
    """
    policy_lp = compute_token_log_probs(policy_logits, labels)

    # Group-relative advantage normalization
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Importance sampling ratio
    if old_log_probs is not None:
        ratio = torch.exp(policy_lp - old_log_probs)
    else:
        ratio = torch.ones_like(policy_lp)

    # PPO-style clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL penalty against reference model
    kl_penalty = torch.tensor(0.0, device=policy_logits.device)
    if reference_log_probs is not None:
        kl_penalty = (policy_lp - reference_log_probs).mean()

    loss = policy_loss + beta * kl_penalty

    return loss, advantages.mean().detach()
