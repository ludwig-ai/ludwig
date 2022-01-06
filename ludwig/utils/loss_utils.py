import torch


def rmspe_loss(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Root mean square percentage error."""
    loss = torch.sqrt(torch.mean(((targets - predictions).float() / targets) ** 2))
    return loss


def mean_confidence_penalty(probabilities: torch.Tensor, num_classes: int) -> torch.Tensor:
    max_entropy = torch.log(torch.tensor(num_classes))
    # clipping needed for avoiding log(0) = -inf
    entropy_per_class = torch.maximum(-probabilities * torch.log(torch.clamp(probabilities, 1e-10, 1)), 0)
    entropy = torch.sum(entropy_per_class, -1)
    penalty = (max_entropy - entropy) / max_entropy
    return torch.mean(penalty)
