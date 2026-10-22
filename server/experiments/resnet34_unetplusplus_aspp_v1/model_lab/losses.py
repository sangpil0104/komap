from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss supporting ignore_index."""
    def __init__(self, num_classes: int = 4, ignore_index: int = 255, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], target: [B, H, W]
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        probs = F.softmax(logits, dim=1)
        safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
        target_one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Mask out ignored pixels
        valid_float = valid_mask.unsqueeze(1).float()
        probs = probs * valid_float
        target_one_hot = target_one_hot * valid_float

        # Compute dice per class
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        cardinality = torch.sum(probs * probs + target_one_hot * target_one_hot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Only average over classes present in the batch
        class_present = target_one_hot.sum(dim=dims) > 0
        if class_present.any():
            return 1.0 - dice_score[class_present].mean()
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """Combines CrossEntropy and SoftDice loss with deep supervision support."""
    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor | list[torch.Tensor]],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        # Deep supervision auxiliary losses if present
        if "aux_logits" in outputs and isinstance(outputs["aux_logits"], list):
            aux_total = 0.0
            for aux in outputs["aux_logits"]:
                aux_ce = self.ce(aux, targets)
                aux_dice = self.dice(aux, targets)
                aux_total = aux_total + (self.ce_weight * aux_ce + self.dice_weight * aux_dice)
            total_loss = total_loss + 0.25 * (aux_total / len(outputs["aux_logits"]))

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "dice_loss": dice_loss.detach(),
        }

