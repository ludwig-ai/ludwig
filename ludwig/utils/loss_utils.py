import torch


def rmspe_loss(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """ Root mean square loss. """
    loss = torch.sqrt(torch.mean(
        ((targets - predictions).float() / targets) ** 2
    ))

    return loss
