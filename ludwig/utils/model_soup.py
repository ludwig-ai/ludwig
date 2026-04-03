"""Model soup: average multiple checkpoint weights for better generalization.

Based on Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time", ICML 2022.
"""

import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


def uniform_soup(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average all state dicts uniformly.

    Args:
        state_dicts: List of model state dicts to average.

    Returns:
        Averaged state dict.
    """
    if not state_dicts:
        raise ValueError("No state dicts provided for model soup")
    if len(state_dicts) == 1:
        return state_dicts[0]

    avg = {}
    n = len(state_dicts)
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts if key in sd])
        avg[key] = (stacked.sum(dim=0) / n).to(state_dicts[0][key].dtype)
    return avg


def greedy_soup(
    state_dicts: list[dict[str, torch.Tensor]],
    model: torch.nn.Module,
    eval_fn: Callable[[], float],
    is_minimize: bool,
) -> dict[str, torch.Tensor]:
    """Greedily add checkpoints that improve validation metric.

    Args:
        state_dicts: List of state dicts sorted by individual performance (best first).
        model: The model to load state dicts into for evaluation.
        eval_fn: Callable that evaluates current model and returns metric value.
        is_minimize: If True, lower metric is better.

    Returns:
        Best soup state dict found.
    """
    if not state_dicts:
        raise ValueError("No state dicts provided for model soup")
    if len(state_dicts) == 1:
        return state_dicts[0]

    # Start with the best individual checkpoint
    ingredients = [state_dicts[0]]
    model.load_state_dict(state_dicts[0], strict=False)
    best_metric = eval_fn()
    best_sd = state_dicts[0]
    logger.info(f"Model soup: starting with best checkpoint, metric={best_metric:.6f}")

    for i, sd in enumerate(state_dicts[1:], 1):
        candidate = uniform_soup(ingredients + [sd])
        model.load_state_dict(candidate, strict=False)
        metric = eval_fn()

        improved = (is_minimize and metric < best_metric) or (not is_minimize and metric > best_metric)
        if improved:
            ingredients.append(sd)
            best_sd = candidate
            best_metric = metric
            logger.info(f"Model soup: added checkpoint {i}, metric improved to {best_metric:.6f}")
        else:
            logger.info(f"Model soup: skipped checkpoint {i}, metric={metric:.6f} (no improvement)")

    logger.info(f"Model soup: final soup uses {len(ingredients)} of {len(state_dicts)} checkpoints")
    return best_sd
