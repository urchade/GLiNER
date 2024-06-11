import torch
import torch.nn.functional as F


def focal_loss_with_logits(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
        label_smoothing: float = 0.0,
        ignore_index: int = -100  # default value for ignored index
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
        label_smoothing (float): Specifies the amount of smoothing when computing the loss, 
                                                                where 0.0 means no smoothing.
        ignore_index (int): Specifies a target value that is ignored and does not contribute
                            to the input gradient. Default: ``-100``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Create a mask to ignore specified index
    valid_mask = targets != ignore_index
    
    # Apply label smoothing if needed
    if label_smoothing != 0:
        with torch.no_grad():
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    # Apply sigmoid activation to inputs
    p = torch.sigmoid(inputs)

    # Compute the binary cross-entropy loss without reduction
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Apply the valid mask to the loss
    loss = loss * valid_mask

    # Apply focal loss modulation if gamma is greater than 0
    if gamma > 0:
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting if alpha is specified
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply reduction method
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.sum() / valid_mask.sum()  # Normalize by the number of valid (non-ignored) elements
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid value for argument 'reduction': '{reduction}'. "
            f"Supported reduction modes: 'none', 'mean', 'sum'"
        )