
import torch, torch.nn as nn, torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()
        intersection = (y_pred * y_true).sum(dim=(2,3))
        union = y_pred.sum(dim=(2,3)) + y_true.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true):
        bce = self.bce(y_pred, y_true.float())
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()

class ComboLoss(nn.Module):
    """ BCEWithLogits + Dice (+ optional Focal) """
    def __init__(self, use_focal=False, focal_alpha=0.25, focal_gamma=2.0,
                 dice_weight=1.0, bce_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.use_focal = use_focal
        self.focal = FocalBCEWithLogits(focal_alpha, focal_gamma)
        self.dw, self.bw, self.fw = dice_weight, bce_weight, focal_weight

    def forward(self, y_pred, y_true):
        loss = self.bw * self.bce(y_pred, y_true.float()) +                self.dw * self.dice(y_pred, y_true)
        if self.use_focal:
            loss += self.fw * self.focal(y_pred, y_true)
        return loss
