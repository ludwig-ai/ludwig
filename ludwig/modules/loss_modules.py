# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import HuberLoss as _HuberLoss
from torch.nn import L1Loss
from torch.nn import MSELoss as _MSELoss
from torchmetrics.functional import mean_absolute_percentage_error

import ludwig.utils.loss_utils as utils
from ludwig.constants import LOGITS
from ludwig.modules.loss_implementations.corn import corn_loss
from ludwig.schema.features.loss.loss import (
    BaseLossConfig,
    BWCEWLossConfig,
    CORNLossConfig,
    DeepSADLossConfig,
    DeepSVDDLossConfig,
    DiceLossConfig,
    DROCCLossConfig,
    Entmax15LossConfig,
    EntropicOpenSetLossConfig,
    FocalLossConfig,
    HuberLossConfig,
    LovaszSoftmaxLossConfig,
    MAELossConfig,
    MAPELossConfig,
    MSELossConfig,
    NextTokenSoftmaxCrossEntropyLossConfig,
    NTXentLossConfig,
    ObjectosphereLossConfig,
    PolyLossConfig,
    RMSELossConfig,
    RMSPELossConfig,
    SequenceSoftmaxCrossEntropyLossConfig,
    SigmoidCrossEntropyLossConfig,
    SoftmaxCrossEntropyLossConfig,
    SparsemaxLossConfig,
)
from ludwig.utils import strings_utils
from ludwig.utils.entmax.losses import Entmax15Loss as _Entmax15Loss
from ludwig.utils.entmax.losses import SparsemaxLoss as _SparsemaxLoss
from ludwig.utils.registry import Registry

# used for Laplace smoothing for candidate samplers
EPSILON = 1.0e-10

loss_impl_registry = Registry[type[nn.Module]]()


def register_loss(config_cls: type[BaseLossConfig]):
    def wrap(cls: type[nn.Module]):
        loss_impl_registry[config_cls] = cls
        return cls

    return wrap


def create_loss(config: BaseLossConfig) -> nn.Module:
    return loss_impl_registry[type(config)](config)


class LogitsInputsMixin:
    @classmethod
    def get_loss_inputs(cls):
        """Maps loss to the desired predicted input type."""
        return LOGITS


@register_loss(MSELossConfig)
class MSELoss(_MSELoss, LogitsInputsMixin):
    """Mean squared error."""

    def __init__(self, config: MSELossConfig):
        super().__init__()


@register_loss(MAELossConfig)
class MAELoss(L1Loss, LogitsInputsMixin):
    """Mean absolute error."""

    def __init__(self, config: MAELossConfig):
        super().__init__()


@register_loss(MAPELossConfig)
class MAPELoss(nn.Module, LogitsInputsMixin):
    """Mean absolute error."""

    def __init__(self, config: MAPELossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return mean_absolute_percentage_error(preds, target)


@register_loss(RMSELossConfig)
class RMSELoss(nn.Module, LogitsInputsMixin):
    """Root mean square error."""

    def __init__(self, config: RMSELossConfig):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(self.mse(preds, target))


@register_loss(RMSPELossConfig)
class RMSPELoss(nn.Module, LogitsInputsMixin):
    """Root mean square percentage error."""

    def __init__(self, config: RMSPELossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        loss = utils.rmspe_loss(target, preds)
        return loss


@register_loss(BWCEWLossConfig)
class BWCEWLoss(nn.Module, LogitsInputsMixin):
    """Binary weighted cross entropy loss."""

    def __init__(self, config: BWCEWLossConfig):
        super().__init__()
        if config.positive_class_weight:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([config.positive_class_weight]))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=config.positive_class_weight)
        self.robust_lambda = config.robust_lambda
        self.confidence_penalty = config.confidence_penalty

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        train_loss = self.loss_fn(preds, target.float())
        # robust lambda
        if self.robust_lambda > 0:
            train_loss = (1 - self.robust_lambda) * train_loss + self.robust_lambda / 2

        train_mean_loss = torch.mean(train_loss)

        # confidence penalty
        if self.confidence_penalty > 0:
            probabilities = torch.sigmoid(preds)
            mean_penalty = utils.mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty

        return train_mean_loss


@register_loss(SoftmaxCrossEntropyLossConfig)
class SoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: SoftmaxCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x num_classes]
                          or shape [batch x num_classes x H x W]
            target: Tensor of shape [batch], where each element is integral
                between 0 and num_classes.
                           or shape [batch x H x W], where each element is integral
                between 0 and num_classes.
        """
        if len(target.shape) == 1 or len(target.shape) == 3:
            # Assumes we are providing the target as a single class, rather than a distribution
            # The target shape can be a 3D tensor [batch x H x W], for image segmentation
            target = target.long()
        return self.loss_fn(preds, target)


@register_loss(SequenceSoftmaxCrossEntropyLossConfig)
class SequenceSoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: SequenceSoftmaxCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.Tensor(config.class_weights), ignore_index=strings_utils.SpecialSymbol.PADDING.value
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=strings_utils.SpecialSymbol.PADDING.value)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x sequence_length x vocab_size]
            target: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.
        """
        target = target.long()
        return self.loss_fn(preds[1:].view(-1, preds.size(-1)), target[1:].view(-1))


@register_loss(NextTokenSoftmaxCrossEntropyLossConfig)
class NextTokenSoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: NextTokenSoftmaxCrossEntropyLossConfig):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x sequence_length x vocab_size]
            target: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.

        Reference implementation:
        https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bert/modeling_bert.py#LL1253C1-L1260C1 # noqa
        """
        target = target.long()
        _, _, vocab_size = preds.shape
        # logits for all tensors except n+1 since each logit tensor at position i represents the log probabilities for
        # the next token i+1 if we were to do argmax on the logits ensor at position i.
        shifted_predictions = preds[:, :-1, :]
        # Shift by 1 since the logits at position 0 in predictions represent the log likelihood of target token 1
        shifted_targets = target[:, 1:]
        return self.loss_fn(shifted_predictions.reshape(-1, vocab_size), shifted_targets.reshape(-1))


@register_loss(SigmoidCrossEntropyLossConfig)
class SigmoidCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: SigmoidCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.ndim != 2:
            raise RuntimeError("SigmoidCrossEntropyLoss currently only supported for 2D tensors.")

        return self.loss_fn(preds.type(torch.float32), target.type(torch.float32))


@register_loss(HuberLossConfig)
class HuberLoss(_HuberLoss, LogitsInputsMixin):
    """Huber loss."""

    def __init__(self, config: HuberLossConfig):
        super().__init__(delta=config.delta)


@register_loss(CORNLossConfig)
class CORNLoss(nn.Module, LogitsInputsMixin):
    """CORN loss."""

    def __init__(self, config: CORNLossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        num_classes = preds.shape[1]
        return corn_loss(preds, target, num_classes=num_classes)


class AnomalyScoreInputsMixin:
    """Mixin for anomaly detection losses: consume anomaly_score tensor (not logits)."""

    @classmethod
    def get_loss_inputs(cls):
        from ludwig.constants import ANOMALY_SCORE

        return ANOMALY_SCORE


@register_loss(DeepSVDDLossConfig)
class DeepSVDDLoss(nn.Module, AnomalyScoreInputsMixin):
    """Deep SVDD loss.

    Hard-boundary: L = mean(||z - c||^2)
    Soft-boundary (nu > 0): L = R + (1/nu) * mean(max(0, dist_sq - R))

    Reference: Ruff et al., ICML 2018.
    """

    def __init__(self, config: DeepSVDDLossConfig):
        super().__init__()
        self.nu = config.nu

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        dist_sq = preds
        if self.nu > 0:
            R = torch.quantile(dist_sq.detach(), 1.0 - self.nu).clamp(min=0.0)
            loss = R + (1.0 / self.nu) * torch.mean(torch.clamp(dist_sq - R, min=0.0))
        else:
            loss = torch.mean(dist_sq)
        return loss


@register_loss(DeepSADLossConfig)
class DeepSADLoss(nn.Module, AnomalyScoreInputsMixin):
    """Deep SAD loss (semi-supervised).

    Normal/unlabeled (target != 1): L_i = ||z - c||^2
    Labeled anomalies (target == 1): L_i = eta / (||z - c||^2 + eps)

    Reference: Ruff et al., ICLR 2020.
    """

    EPSILON = 1e-6

    def __init__(self, config: DeepSADLossConfig):
        super().__init__()
        self.eta = config.eta

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        dist_sq = preds
        target = target.float()
        is_anomaly = (target == 1).float()
        normal_loss = dist_sq * (1.0 - is_anomaly)
        anomaly_loss = self.eta / (dist_sq + self.EPSILON) * is_anomaly
        return torch.mean(normal_loss + anomaly_loss)


@register_loss(DROCCLossConfig)
class DROCCLoss(nn.Module, AnomalyScoreInputsMixin):
    """DROCC loss: adversarial regularisation to prevent hypersphere collapse.

    Combines SVDD objective with hinge loss on score-space perturbations.

    Reference: Goyal et al., ICML 2020.
    """

    def __init__(self, config: DROCCLossConfig):
        super().__init__()
        self.perturbation_strength = config.perturbation_strength

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        dist_sq = preds
        svdd_loss = torch.mean(dist_sq)
        with torch.no_grad():
            noise_scale = self.perturbation_strength * dist_sq.detach().std().clamp(min=1e-6)
            perturbed = dist_sq.detach() + noise_scale * torch.randn_like(dist_sq)
        hinge = torch.mean(torch.clamp(dist_sq - perturbed, min=0.0))
        return svdd_loss + self.perturbation_strength * hinge


@register_loss(EntropicOpenSetLossConfig)
class EntropicOpenSetLoss(nn.Module, LogitsInputsMixin):
    """Entropic Open-Set Loss from Dhamija et al., NeurIPS 2018.

    For known-class samples (target != background_class):
        L = CrossEntropy(logits, target)

    For unknown/background samples (target == background_class):
        L = sum_i( p_i * log(p_i) )   # negative entropy → maximise entropy

    Without a background_class this reduces to standard cross-entropy.

    Reference: https://arxiv.org/abs/1811.04110
    """

    def __init__(self, config: EntropicOpenSetLossConfig):
        super().__init__()
        self.background_class = config.background_class

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        logits = preds[LOGITS] if isinstance(preds, dict) else preds

        if self.background_class is None:
            return F.cross_entropy(logits, target)

        known_mask = target != self.background_class
        unknown_mask = ~known_mask

        loss = logits.new_tensor(0.0)

        if known_mask.any():
            loss = loss + F.cross_entropy(logits[known_mask], target[known_mask])

        if unknown_mask.any():
            probs = torch.softmax(logits[unknown_mask], dim=-1)
            # Negative entropy: p * log(p). Minimising this maximises H(p).
            loss = loss + (probs * torch.log(probs + EPSILON)).sum(dim=-1).mean()

        return loss


@register_loss(ObjectosphereLossConfig)
class ObjectosphereLoss(nn.Module, LogitsInputsMixin):
    """Objectosphere Loss from Dhamija et al., NeurIPS 2018.

    For known-class samples:
        L = CrossEntropy(logits, target) + hinge(||logits|| - xi)

    For unknown/background samples:
        L = NegEntropy(logits) + zeta * ||logits||^2

    The hinge term for known samples is max(0, xi - ||z||)^2, pushing logit
    norms above xi. The magnitude term for unknowns pulls norms toward zero,
    so out-of-distribution inputs produce near-uniform, low-magnitude outputs
    that are easy to detect with a simple norm threshold at inference time.

    Without a background_class this reduces to CE + known-class hinge only.

    Reference: https://arxiv.org/abs/1811.04110
    """

    def __init__(self, config: ObjectosphereLossConfig):
        super().__init__()
        self.background_class = config.background_class
        self.xi = config.xi
        self.zeta = config.zeta

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        logits = preds[LOGITS] if isinstance(preds, dict) else preds

        if self.background_class is None:
            # No unknowns: CE + magnitude push for all samples.
            ce = F.cross_entropy(logits, target)
            mag = logits.norm(dim=-1)
            hinge = torch.clamp(self.xi - mag, min=0.0).pow(2).mean()
            return ce + hinge

        known_mask = target != self.background_class
        unknown_mask = ~known_mask

        loss = logits.new_tensor(0.0)

        if known_mask.any():
            known_logits = logits[known_mask]
            ce = F.cross_entropy(known_logits, target[known_mask])
            mag = known_logits.norm(dim=-1)
            hinge = torch.clamp(self.xi - mag, min=0.0).pow(2).mean()
            loss = loss + ce + hinge

        if unknown_mask.any():
            unknown_logits = logits[unknown_mask]
            probs = torch.softmax(unknown_logits, dim=-1)
            neg_entropy = (probs * torch.log(probs + EPSILON)).sum(dim=-1).mean()
            mag_penalty = unknown_logits.norm(dim=-1).pow(2).mean()
            loss = loss + neg_entropy + self.zeta * mag_penalty

        return loss


@register_loss(FocalLossConfig)
class FocalLoss(nn.Module, LogitsInputsMixin):
    """Focal Loss for classification with class imbalance.

    Applies a modulating factor ``(1 - p_t)^gamma`` to the standard
    cross-entropy loss so that easy examples contribute less to the gradient
    and training is focused on hard, misclassified examples.

    Formula::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Supports both binary (scalar logits) and multi-class (logit vectors) inputs.
    For binary inputs, ``alpha_t`` balances positive/negative classes.
    For multi-class inputs, the modulating factor is applied without per-class alpha.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
        https://arxiv.org/abs/1708.02002
    """

    def __init__(self, config: FocalLossConfig):
        super().__init__()
        self.alpha = config.alpha
        self.gamma = config.gamma

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds:  [batch] logits (binary) or [batch x num_classes] logits (multi-class).
            target: [batch] integer class labels (0/1 for binary).
        """
        if preds.ndim == 1:
            # Binary case: compute per-element BCE then reweight with alpha
            bce = F.binary_cross_entropy_with_logits(preds, target.float(), reduction="none")
            p_t = torch.exp(-bce)
            alpha_t = self.alpha * target.float() + (1 - self.alpha) * (1 - target.float())
            focal = alpha_t * (1 - p_t) ** self.gamma * bce
            return focal.mean()
        else:
            # Multi-class case: alpha not applied (symmetric across classes)
            ce = F.cross_entropy(preds, target.long(), reduction="none")
            p_t = torch.exp(-ce)
            focal = (1 - p_t) ** self.gamma * ce
            return focal.mean()


@register_loss(DiceLossConfig)
class DiceLoss(nn.Module, LogitsInputsMixin):
    """Dice Loss for image segmentation.

    Computes one minus the Dice coefficient between predicted soft masks and
    one-hot ground-truth masks.  A ``smooth`` term prevents division by zero
    when both prediction and target are empty.

    Formula::

        Dice = 1 - (2 * sum(p * t) + smooth) / (sum(p) + sum(t) + smooth)

    Inputs are expected as class logits of shape ``[B, C, H, W]`` and integer
    targets of shape ``[B, H, W]``.

    Reference:
        Milletari et al., "V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation", 3DV 2016.
        https://arxiv.org/abs/1606.04797
    """

    def __init__(self, config: DiceLossConfig):
        super().__init__()
        self.smooth = config.smooth

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds:  Float tensor [B, C, H, W] of class logits.
            target: Long tensor  [B, H, W]    of integer class indices.
        """
        num_classes = preds.shape[1]
        probs = F.softmax(preds, dim=1)  # [B, C, H, W]
        # One-hot encode target: [B, H, W] -> [B, C, H, W]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        # Flatten spatial dims for dot-product computation
        probs_flat = probs.contiguous().view(probs.shape[0], num_classes, -1)  # [B, C, N]
        target_flat = target_one_hot.contiguous().view(target_one_hot.shape[0], num_classes, -1)
        intersection = (probs_flat * target_flat).sum(dim=2)  # [B, C]
        union = probs_flat.sum(dim=2) + target_flat.sum(dim=2)  # [B, C]
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coeff.mean()


@register_loss(LovaszSoftmaxLossConfig)
class LovaszSoftmaxLoss(nn.Module, LogitsInputsMixin):
    """Lovasz-Softmax Loss for multi-class semantic segmentation.

    Uses the Lovasz extension of submodular functions to construct a convex
    surrogate for the per-class intersection-over-union (IoU) loss.  Unlike
    the Dice loss, it directly targets the mIoU metric used in segmentation
    benchmarks.

    Inputs are expected as class logits of shape ``[B, C, H, W]`` and integer
    targets of shape ``[B, H, W]``.

    Reference:
        Berman et al., "The Lovasz-Softmax Loss: A Tractable Surrogate for
        the Optimization of the Intersection-Over-Union Measure in Neural
        Networks", CVPR 2018.
        https://arxiv.org/abs/1705.08790
    """

    def __init__(self, config: LovaszSoftmaxLossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds:  Float tensor [B, C, H, W] of class logits.
            target: Long tensor  [B, H, W]    of integer class indices.
        """
        num_classes = preds.shape[1]
        probas = F.softmax(preds, dim=1)  # [B, C, H, W]
        return self._lovasz_softmax(probas, target, num_classes)

    @staticmethod
    def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
        """Compute the Lovasz extension coefficients from a sorted ground-truth vector."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def _lovasz_softmax_flat(self, probas: Tensor, labels: Tensor, num_classes: int) -> Tensor:
        """Compute the Lovasz-Softmax loss on pixel-flattened tensors."""
        if probas.numel() == 0:
            return probas * 0.0
        loss = torch.zeros(1, device=probas.device, dtype=probas.dtype)
        for c in range(num_classes):
            fg = (labels == c).float()  # foreground indicator for class c
            if fg.sum() == 0:
                continue
            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            gt_sorted = fg[perm]
            grad = self._lovasz_grad(gt_sorted)
            loss += torch.dot(errors_sorted, grad)
        return loss / num_classes

    def _lovasz_softmax(self, probas: Tensor, labels: Tensor, num_classes: int) -> Tensor:
        B, C, H, W = probas.shape
        probas_flat = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        labels_flat = labels.view(-1)  # [B*H*W]
        return self._lovasz_softmax_flat(probas_flat, labels_flat, num_classes)


@register_loss(NTXentLossConfig)
class NTXentLoss(nn.Module, LogitsInputsMixin):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    The SimCLR objective.  Given a batch of ``N`` vector representations, the
    loss is computed assuming that consecutive pairs ``(2i, 2i+1)`` are positive
    pairs (augmented views of the same example) and all other ``2(N-1)``
    examples in the batch are negatives.

    Formula (for pair (i, j) with temperature tau)::

        L_i = -log(
            exp(sim(z_i, z_j) / tau) /
            sum_{k != i} exp(sim(z_i, z_k) / tau)
        )

    where sim denotes cosine similarity and tau is the temperature.

    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning of Visual
        Representations" (SimCLR), ICML 2020.
        https://arxiv.org/abs/2002.05709
    """

    def __init__(self, config: NTXentLossConfig):
        super().__init__()
        self.temperature = config.temperature

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds:  Float tensor [B, D] of vector embeddings (logits output).
            target: Unused; required for interface compatibility.
        """
        z = F.normalize(preds, dim=1)  # [B, D]
        sim = torch.mm(z, z.T) / self.temperature  # [B, B]
        B = z.shape[0]
        # Mask out self-similarities
        mask = torch.eye(B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))
        # Positive-pair labels: pair (i, j) where j = i XOR 1
        # (even i -> i+1, odd i -> i-1) -- standard SimCLR pairing.
        if B >= 2 and B % 2 == 0:
            labels = torch.arange(B, device=z.device) ^ 1
        else:
            # Degenerate batch: use nearest non-self neighbour as positive
            labels = sim.argmax(dim=1)
        return F.cross_entropy(sim, labels)


@register_loss(PolyLossConfig)
class PolyLoss(nn.Module, LogitsInputsMixin):
    """PolyLoss for multi-class classification.

    Extends cross-entropy with a first-order polynomial correction term
    ``epsilon * (1 - p_t)`` that upweights hard examples where the model
    places low probability on the correct class.

    Formula::

        PolyLoss = CE(p_t) + epsilon * (1 - p_t)

    where ``p_t`` is the predicted softmax probability of the ground-truth class.

    Reference:
        Leng et al., "PolyLoss: A Polynomial Expansion Perspective of
        Classification Loss Functions", ICLR 2022.
        https://arxiv.org/abs/2204.12511
    """

    def __init__(self, config: PolyLossConfig):
        super().__init__()
        self.epsilon = config.epsilon
        self.ce_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds:  Float tensor [B, C] of class logits.
            target: Long tensor  [B]    of integer class indices.
        """
        target = target.long()
        ce = self.ce_fn(preds, target)  # [B]
        probs = F.softmax(preds, dim=-1)  # [B, C]
        p_t = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)  # [B]
        poly = ce + self.epsilon * (1.0 - p_t)
        return poly.mean()


@register_loss(SparsemaxLossConfig)
class SparsemaxLoss(nn.Module, LogitsInputsMixin):
    """Sparsemax Loss: a sparse alternative to softmax cross-entropy.

    The natural loss companion to the sparsemax activation, derived as the
    Fenchel conjugate of the sparsemax Omega function.  Assigns zero gradient
    to classes outside the sparsemax support, producing exact sparsity in the
    probability simplex.

    Reference:
        Martins & Astudillo, "From Softmax to Sparsemax: A Sparse Model of
        Attention and Multi-Label Classification", ICML 2016.
        https://arxiv.org/abs/1602.02068
    """

    def __init__(self, config: SparsemaxLossConfig):
        super().__init__()
        self._loss_fn = _SparsemaxLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds:  Float tensor [B, C] of class logits.
            target: Long tensor  [B]    of integer class indices.
        """
        return self._loss_fn(preds, target.long())


@register_loss(Entmax15LossConfig)
class Entmax15Loss(nn.Module, LogitsInputsMixin):
    """Entmax-1.5 Loss: a semi-sparse alternative to softmax cross-entropy.

    The Fenchel-conjugate loss of the alpha=1.5 entmax activation.  Produces
    moderately sparse probability distributions that lie between softmax
    (dense) and sparsemax (maximally sparse).

    Reference:
        Peters et al., "Sparse Sequence-to-Sequence Models", ACL 2019.
        https://arxiv.org/abs/1905.05702
    """

    def __init__(self, config: Entmax15LossConfig):
        super().__init__()
        self._loss_fn = _Entmax15Loss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds:  Float tensor [B, C] of class logits.
            target: Long tensor  [B]    of integer class indices.
        """
        return self._loss_fn(preds, target.long())
