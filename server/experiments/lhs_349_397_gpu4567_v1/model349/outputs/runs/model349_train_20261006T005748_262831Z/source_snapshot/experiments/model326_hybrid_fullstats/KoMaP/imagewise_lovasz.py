"""Memory-bounded present-class Lovasz-Softmax for per-image macro IoU."""

import torch
import torch.nn.functional as F


def _lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    total = sorted_foreground.sum()
    intersection = total - sorted_foreground.cumsum(0)
    union = total + (1.0 - sorted_foreground).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-7)
    if gradient.numel() > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


def imagewise_lovasz_loss(logits: torch.Tensor, target: torch.Tensor,
                          max_pixels: int = 131072) -> torch.Tensor:
    """Average present-class Lovasz loss per image.

    Uniform sampling with replacement bounds the sorting cost on large crops;
    the class loss is still averaged per image and only over present classes.
    """
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0]:
        raise ValueError("expected logits [B,C,H,W] and target [B,H,W]")
    if logits.shape[1] < 2 or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("logit/target spatial shapes or class count are invalid")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")

    probabilities = F.softmax(logits.float(), dim=1)
    image_losses = []
    for batch_index in range(target.shape[0]):
        labels = target[batch_index].reshape(-1)
        class_probabilities = probabilities[batch_index].permute(1, 2, 0).reshape(
            -1, logits.shape[1])
        if labels.numel() > max_pixels:
            indices = torch.randint(labels.numel(), (max_pixels,), device=labels.device)
            labels = labels[indices]
            class_probabilities = class_probabilities[indices]

        class_losses = []
        for class_id in torch.unique(labels):
            foreground = (labels == class_id).to(probabilities.dtype)
            errors = (foreground - class_probabilities[:, class_id]).abs()
            sorted_errors, permutation = torch.sort(errors, descending=True)
            class_losses.append(torch.dot(
                sorted_errors, _lovasz_gradient(foreground[permutation])))
        if class_losses:
            image_losses.append(torch.stack(class_losses).mean())
    if not image_losses:
        return probabilities.sum() * 0.0
    return torch.stack(image_losses).mean()
