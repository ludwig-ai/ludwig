# Source: https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
# Sebastian Raschka 2020-2021
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch
import torch.nn.functional as F


def coral_loss(logits, levels, importance_weights=None, reduction="mean"):
    """Computes the CORAL loss described in.

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CORAL layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `coral_pytorch.dataset.levels_from_labelbatch`).

    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import coral_loss
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> coral_loss(logits, levels)
    tensor(0.6920)
    """

    if not logits.shape == levels.shape:
        raise ValueError(
            "Please ensure that logits ({}) has the same shape as levels ({}). ".format(logits.shape, levels.shape)
        )

    term1 = F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels)

    if importance_weights is not None:
        term1 *= importance_weights

    val = -torch.sum(term1, dim=1)

    if reduction == "mean":
        loss = torch.mean(val)
    elif reduction == "sum":
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = 'Invalid value for `reduction`. Should be "mean", ' '"sum", or None. Got %s' % reduction
        raise ValueError(s)

    return loss


def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss described in our forthcoming 'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities' manuscript.

    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.

    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.

    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    sets = []
    for i in range(num_classes - 1):
        label_mask = y_train > i - 1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.0
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred) * train_labels + (F.logsigmoid(pred) - pred) * (1 - train_labels))
        losses += loss

    return losses / num_examples
