# utils.py
import numpy as np
import torch
from config import Config
def calculate_iou(preds, labels):
    """计算mIoU"""
    preds = torch.argmax(preds, dim=1)
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    ious = []
    for cls in range(Config.num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(union))
    return np.nanmean(ious)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def pixel_accuracy(output, mask):
    _, preds = torch.max(output, 1)
    valid = (mask >= 0)
    acc = torch.sum((preds == mask) * valid) / torch.sum(valid)
    return acc.item()