import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        Focal Loss for addressing class imbalance in classification problems.

        Args:
        - gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
        - alpha (Tensor or None): Class weights for addressing class imbalance. Should have size [num_classes].
        - reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
        - inputs (Tensor): Logits from the model of shape [batch_size, num_classes].
        - targets (Tensor): Ground truth labels of shape [batch_size].
        
        Returns:
        - Tensor: Computed focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Cross-entropy loss for each instance
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss *= alpha_t

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
