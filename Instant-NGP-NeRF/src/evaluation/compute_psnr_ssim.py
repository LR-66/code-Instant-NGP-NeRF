"""Evaluation script for PSNR and SSIM metrics."""

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np


def compute_psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute PSNR.
    
    PSNR = 10 * log10(I_Z^2 / MSE)
    """
    mse = F.mse_loss(rendered, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def compute_ssim(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute SSIM.
    
    SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x^2 + μ_y^2 + C1)(σ_x^2 + σ_y^2 + C2))
    """
    # Convert to numpy and normalize
    rendered_np = rendered.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().transpose(1, 2, 0)
    
    ssim_value = ssim(rendered_np, target_np, channel_axis=2, data_range=1.0)
    return ssim_value


def compute_lpips(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute LPIPS using pre-trained network.
    """
    import lpips
    model = lpips.LPIPS(net='alex')
    with torch.no_grad():
        score = model(rendered.unsqueeze(0), target.unsqueeze(0))
    return score.item()


def compute_chamfer_distance(points_pred: torch.Tensor, points_gt: torch.Tensor) -> float:
    """
    Compute Chamfer distance between two point clouds.
    
    CD(S1, S2) = (1/|S1|) Σ_{p∈S1} min_{q∈S2} ||p-q||² + (1/|S2|) Σ_{q∈S2} min_{p∈S1} ||q-p||²
    """
    # Compute pairwise distances
    dist = torch.cdist(points_pred, points_gt)  # (N1, N2)
    
    # Chamfer distance
    cd1 = dist.min(dim=1)[0].mean()
    cd2 = dist.min(dim=0)[0].mean()
    
    return (cd1 + cd2).item()


def evaluate_viewpoint(model, dataset, view_idx):
    """Evaluate a single viewpoint."""
    data = dataset[view_idx]
    camera_matrix = data['camera_matrix']
    
    # Render image
    rendered = model.render(camera_matrix)
    target = data['image']
    
    psnr = compute_psnr(rendered, target)
    ssim_val = compute_ssim(rendered, target)
    lpips_val = compute_lpips(rendered, target)
    
    return {
        'psnr': psnr,
        'ssim': ssim_val,
        'lpips': lpips_val,
    }