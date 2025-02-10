import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """
    if pred.dim() > target.dim():
        pred = torch.argmax(pred, dim=1)  # Convert to class indices
    assert pred.shape == target.shape
    
    ious = []
    
    for cls in range(n_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls) - intersection
        
        if union == 0:
            continue  # Skip if class not present
            
        iou = intersection / union
        ious.append(iou)
    
    if not ious:
        return 0.0
        
    return torch.stack(ious).mean().item()

def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """
    assert pred.shape == target.shape
    correct = torch.sum(pred == target).item()
    total = pred.numel()
    return correct / total