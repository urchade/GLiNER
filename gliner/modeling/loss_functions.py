import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    prob_margin: float = 0.0,
    reduction: str = "none",
    label_smoothing: float = 0.0,
    normalize_prob: bool = True,
    ignore_index: int = -100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute focal loss for binary classification with logits.

    Focal loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples. Originally proposed in
    RetinaNet (https://arxiv.org/abs/1708.02002) for dense object detection.

    The focal loss is defined as:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t is the model's estimated probability for the correct class.

    Args:
        inputs (torch.Tensor): Predicted logits of arbitrary shape.
            The predictions for each example (not probabilities).
        targets (torch.Tensor): Ground truth binary labels with the same
            shape as inputs. Should contain 0 for negative class and 1
            for positive class.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Set to -1 to disable. The weight
            for positive examples is alpha, and for negative examples is
            (1 - alpha). Defaults to 0.25.
        gamma (float, optional): Exponent of the modulating factor (1 - p_t)
            to balance easy vs hard examples. Higher values give more weight
            to hard examples. Set to 0 to disable focal modulation.
            Defaults to 2.
        prob_margin (float, optional): Margin to subtract from predicted
            probabilities for negative examples. Can help with hard negative
            mining. Defaults to 0.
        reduction (str, optional): Specifies the reduction to apply to the
            output. Options are:
            - 'none': No reduction, return loss for each element
            - 'mean': Return the mean loss (normalized by valid elements)
            - 'sum': Return the sum of all losses
            Defaults to 'none'.
        label_smoothing (float, optional): Amount of label smoothing to apply.
            Targets become: target * (1 - label_smoothing) + 0.5 * label_smoothing.
            Value should be in range [0, 1]. Defaults to 0.0 (no smoothing).
        normalize_prob (bool, optional): If True, apply sigmoid to inputs to
            get probabilities. If False, treat inputs as probabilities directly.
            Defaults to True.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the loss gradient. Defaults to -100.
        eps (float, optional): Small epsilon value for numerical stability
            when computing logarithms. Defaults to 1e-6.

    Returns:
        torch.Tensor: Loss tensor. Shape depends on reduction parameter:
            - If reduction='none': same shape as inputs
            - If reduction='mean' or 'sum': scalar tensor

    Raises:
        ValueError: If an invalid reduction mode is specified.

    Example:
        >>> inputs = torch.randn(32, 100)  # logits
        >>> targets = torch.randint(0, 2, (32, 100)).float()
        >>> loss = focal_loss_with_logits(inputs, targets, reduction="mean")
    """
    # Create a mask to ignore specified index
    valid_mask = targets != ignore_index

    # Apply label smoothing if needed
    if label_smoothing != 0:
        with torch.no_grad():
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    # Apply sigmoid activation to inputs
    if normalize_prob:
        p = torch.sigmoid(inputs)
    else:
        p = inputs

    pm = torch.clamp(p - prob_margin, max=1.0)

    # Compute the binary cross-entropy loss without reduction
    pos_term = -targets * torch.log(p.clamp(min=eps))
    neg_term = -(1.0 - targets) * torch.log((1.0 - pm).clamp(min=eps))
    loss = pos_term + neg_term

    # Apply the valid mask to the loss
    loss = loss * valid_mask

    # Apply focal loss modulation if gamma is greater than 0
    if gamma > 0:
        p_t = p * targets + (1 - pm) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting if alpha is specified
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply reduction method
    if reduction == "none":
        return loss
    elif reduction == "mean":
        # Normalize by the number of valid (non-ignored) elements
        return loss.sum() / valid_mask.sum()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid value for argument 'reduction': '{reduction}'. Supported reduction modes: 'none', 'mean', 'sum'"
        )


def cross_entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "sum",
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute cross-entropy loss for multi-class classification.

    A wrapper around PyTorch's F.cross_entropy that reshapes inputs and
    targets for convenient use in GLiNER models. This function flattens
    batch and sequence dimensions before computing the loss.

    Args:
        inputs (torch.Tensor): Predicted logits of shape [..., num_classes].
            Typically shape [batch_size, seq_len, num_classes] or similar.
            Will be reshaped to [-1, num_classes] internally.
        targets (torch.Tensor): Ground truth class indices with shape [...].
            Should have one fewer dimension than inputs (no class dimension).
            Values should be in range [0, num_classes - 1] or equal to
            ignore_index. Will be reshaped to [-1] internally.
        reduction (str, optional): Specifies the reduction to apply to the
            output. Options are:
            - 'none': No reduction, return loss for each element
            - 'mean': Return the mean loss
            - 'sum': Return the sum of all losses
            Defaults to 'sum'.
        label_smoothing (float, optional): Amount of label smoothing to apply.
            Value should be in range [0, 1]. Defaults to 0.0 (no smoothing).
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the loss gradient. Defaults to -100.

    Returns:
        torch.Tensor: Loss tensor. Shape depends on reduction parameter:
            - If reduction='none': shape [batch_size * seq_len] (flattened)
            - If reduction='mean' or 'sum': scalar tensor

    Example:
        >>> inputs = torch.randn(8, 50, 10)  # batch=8, seq_len=50, classes=10
        >>> targets = torch.randint(0, 10, (8, 50))  # class indices
        >>> loss = cross_entropy_loss(inputs, targets, reduction="mean")
    """
    cls_size = inputs.shape[-1]
    inputs = inputs.reshape(-1, cls_size)
    targets = targets.reshape(-1)

    loss = F.cross_entropy(
        inputs, targets, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction
    )

    return loss
