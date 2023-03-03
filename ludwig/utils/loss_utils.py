import torch


def rmspe_loss(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Root mean square percentage error.

    Bad predictions can lead to arbitrarily large RMSPE values, especially if some values of targets are very close to
    zero. We return a large value instead of inf when (some) targets are zero.
    """
    epsilon = 1e-4
    # add epsilon if targets are zero to avoid division by zero
    denominator = targets + epsilon * (targets == 0).float()
    loss = torch.sqrt(torch.mean(((targets - predictions).float() / denominator) ** 2))
    return loss


def mean_confidence_penalty(probabilities: torch.Tensor, num_classes: int) -> torch.Tensor:
    max_entropy = torch.log(torch.tensor(num_classes))
    # clipping needed for avoiding log(0) = -inf
    entropy_per_class, _ = torch.max(-probabilities * torch.log(torch.clamp(probabilities, 1e-10, 1)), dim=0)
    entropy = torch.sum(entropy_per_class, -1)
    penalty = (max_entropy - entropy) / max_entropy
    return torch.mean(penalty)
