import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''focal loss for binary classification, reduces to plain BCE when gamma is 0

    down-weights easy, well classified examples so training focuses on the harder ones
    '''
    def __init__(self, gamma: float = 0.0) -> None:
        '''builds the loss with the given focusing parameter

        Parameters
        ----------
        gamma : float, optional
            focusing parameter, by default 0.0 (equivalent to plain BCE)
        '''
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''computes the focal loss between logits and targets

        in experiments when focal gamma is 0, is equivalent to BCE

        Parameters
        ----------
        logits : torch.Tensor
            raw model outputs (pre-sigmoid) shape (B, 1)
        targets : torch.Tensor
            true binary labels shape (B, 1)

        Returns
        -------
        torch.Tensor
            scalar mean loss over the batch
        '''
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # gamma == 0 reduces focal loss to plain BCE
        if self.gamma == 0.0:
            return bce.mean()

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()