import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return sigmoid_focal_loss(
            inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )
