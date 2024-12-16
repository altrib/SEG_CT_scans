import torch.nn as nn
import torch
import sklearn.metrics
import numpy as np


def dice_loss(pred, target, smooth=1):
    pred = torch.softmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(pred, target)
        dice = dice_loss(pred, torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2))
        return ce_loss + dice
    



def pixel_accuracy(preds, targets):
    # preds = preds.argmax(dim=1)  # Convert logits to class predictions
    correct = (preds == targets).sum()
    return correct.float() / targets.numel()


def mean_iou(preds, targets, num_classes):
    # preds = preds.argmax(dim=1)  # Convert logits to class predictions
    ious = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().float()
        union = ((preds == cls) | (targets == cls)).sum().float()
        if union == 0:
            ious.append(torch.tensor(0.0))  # Avoid division by zero
        else:
            ious.append(intersection / union)
    return sum(ious) / len(ious)


def dice_coefficient(preds, targets, num_classes):
    # preds = preds.argmax(dim=1)
    dices = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().float()
        pred_area = (preds == cls).sum().float()
        target_area = (targets == cls).sum().float()
        dice = (2 * intersection) / (pred_area + target_area + 1e-6)  # Add epsilon to avoid division by zero
        dices.append(dice)
    return sum(dices) / len(dices)

def adjusted_rand_index_score(preds,targets):
    individual_rand_index = []
    for i in range(len(targets)): #batch_size
        individual_rand_index.append(sklearn.metrics.adjusted_rand_score(targets[i].flatten(), preds[i].flatten()))
    return np.mean(individual_rand_index)