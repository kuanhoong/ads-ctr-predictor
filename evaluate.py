
import torch
import torch.nn as nn
import cv2
import numpy as np

def compute_saliency_map(model, img_tensor, label, device='cpu'):
    """
    Computes saliency map for a given image.
    img_tensor: (1, C, H, W)
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_()
    
    output = model(img_tensor)
    
    # We want to maximize the output (CTR) or just see what contributes to it?
    # Usually we propagate back from the score.
    # Score is a scalar.
    
    score = output[0]
    score.backward()
    
    # Get gradient
    # Shape: (1, 3, 144, 144)
    grad = img_tensor.grad.data.abs()
    
    # Max across channels to get (1, 144, 144)
    saliency, _ = torch.max(grad, dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    # Normalize 0-1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency

def save_saliency_map(original_img_np, saliency_map, save_path="saliency.png"):
    """
    Overlays saliency map on original image (heatmap).
    original_img_np: (144, 144, 3) uint8 RGB
    """
    # Resize saliency to image size (already matches here, 144x144)
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    
    cam = heatmap + np.float32(original_img_np) / 255.0
    cam = cam / np.max(cam)
    
    cv2.imwrite(save_path, np.uint8(255 * cam))
