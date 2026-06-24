"""Distillation of Gaussian primitives from trained NeRF."""

import os
import sys
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.nerf import NeRF
from models.gaussian_model import GaussianModel
from models.renderer import GaussianRenderer
from optimization.losses import GaussianDistillationLoss


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'distill.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def sample_density_field(nerf_model, device, num_samples: int = 100000):
    """
    Sample the density field of the trained NeRF.
    
    Returns point cloud of locations with non-zero density.
    """
    logging.info("Sampling density field...")
    nerf_model.eval()
    
    # Uniformly sample in 3D space
    samples = torch.rand(num_samples, 3, device=device) * 2 - 1  # [-1, 1]^3
    
    with torch.no_grad():
        density = nerf_model.get_density(samples)
    
    # Keep only valid regions (density > threshold)
    # Calculate threshold as 1.5 * mean density
    valid_mask = density > 1.5 * density.mean()
    valid_points = samples[valid_mask.squeeze(-1)]
    valid_density = density[valid_mask.squeeze(-1)]
    
    logging.info(f"Retained {len(valid_points)} valid points out of {num_samples}")
    return valid_points, valid_density


def cluster_and_initialize_gaussians(points, density, cluster_radius: float = 0.002):
    """
    Cluster points into Gaussian primitives using fixed-radius neighbor search.
    """
    logging.info("Clustering points into Gaussian primitives...")
    # (Simplified clustering implementation)
    
    gaussian_positions = []
    gaussian_opacities = []
    
    # Iterative clustering
    remaining = points.clone()
    remaining_density = density.clone()
    
    while len(remaining) > 0:
        # Pick a seed point
        seed = remaining[0]
        seed_density = remaining_density[0]
        
        # Find neighbors within radius
        distances = torch.norm(remaining - seed, dim=1)
        neighbors = distances < cluster_radius
        
        if neighbors.sum() > 5:  # Minimum cluster size
            # Compute cluster center and average density
            cluster_points = remaining[neighbors]
            cluster_density = remaining_density[neighbors]
            center = cluster_points.mean(dim=0)
            avg_opacity = torch.sigmoid(cluster_density.mean())
            
            gaussian_positions.append(center)
            gaussian_opacities.append(avg_opacity)
        
        # Remove selected points
        remaining = remaining[~neighbors]
        remaining_density = remaining_density[~neighbors]
        
        # Safety limit
        if len(gaussian_positions) >= 200000:
            break
    
    positions = torch.stack(gaussian_positions)
    opacities = torch.stack(gaussian_opacities)
    
    logging.info(f"Initialized {len(positions)} Gaussian primitives")
    return positions, opacities


def initialize_covariances(points, positions):
    """
    Initialize covariance matrices using neighborhood point distribution.
    """
    logging.info("Initializing covariance matrices...")
    covariances = []
    
    for i, pos in enumerate(positions):
        # Find nearby points
        distances = torch.norm(points - pos, dim=1)
        neighbors = points[distances < 0.01]
        
        if len(neighbors) > 10:
            # Compute covariance of local point cloud
            centered = neighbors - neighbors.mean(dim=0)
            cov = (centered.T @ centered) / len(neighbors)
            # Add small regularization
            cov += 1e-6 * torch.eye(3, device=cov.device)
            
            # Convert to 6-DOF representation
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # Use rotation matrix from eigenvectors
            R = eigvecs
            s = torch.sqrt(eigvals.clamp(min=1e-6))
            # 6-DOF: quaternion (4) + scales (3)
            q = matrix_to_quaternion(R)
            cov_flat = torch.cat([q, s])
        else:
            # Default: isotropic small Gaussian
            cov_flat = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01], device=positions.device)
        
        covariances.append(cov_flat)
    
    return torch.stack(covariances)


def matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion."""
    trace = R.trace()
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return torch.stack([w, x, y, z])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config['logging']['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load trained NeRF
    nerf_path = config['input']['checkpoint']
    checkpoint = torch.load(nerf_path, map_location=device)
    
    model = NeRF().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded NeRF from {nerf_path}")
    
    # Sample density field
    points, density = sample_density_field(
        model,
        device,
        num_samples=500000,
    )
    
    # Cluster and initialize Gaussian parameters
    positions, opacities = cluster_and_initialize_gaussians(
        points,
        density,
        cluster_radius=config['distillation']['cluster_radius'],
    )
    
    covariances = initialize_covariances(points, positions)
    sh_coeffs = torch.randn(len(positions), 16, 3, device=device) * 0.01
    
    # Initialize Gaussian model
    gaussian_model = GaussianModel(
        max_primitives=config['distillation']['max_gaussians'],
        sh_degree=3,
    ).to(device)
    
    gaussian_model.positions = positions
    gaussian_model.opacities = opacities
    gaussian_model.covariances = covariances
    gaussian_model.sh_coeffs = sh_coeffs
    
    # Joint optimization
    optimizer = optim.Adam([
        {'params': [gaussian_model.positions], 'lr': 1.6e-4},
        {'params': [gaussian_model.covariances], 'lr': 5.0e-3},
        {'params': [gaussian_model.sh_coeffs], 'lr': 2.5e-3},
    ])
    
    criterion = GaussianDistillationLoss(
        l1_weight=config['loss']['l1_weight'],
        perceptual_weight=config['loss']['perceptual_weight'],
    )
    
    renderer = GaussianRenderer()
    
    logging.info("Starting joint optimization...")
    for step in range(config['distillation']['optimize_steps']):
        optimizer.zero_grad()
        
        # Render from random viewpoint
        camera_matrix = generate_random_camera()
        projection_matrix = generate_projection()
        view_dir = torch.randn(3, device=device)
        view_dir = view_dir / view_dir.norm()
        
        rendered = renderer(gaussian_model, camera_matrix, projection_matrix, view_dir)
        target = get_target_image(rendered, camera_matrix, projection_matrix)
        
        loss = criterion(rendered, target)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            logging.info(f"Step {step+1}: loss={loss.item():.4f}")
    
    # Save Gaussian model
    output_path = config['output']['checkpoint']
    torch.save({
        'model_state_dict': gaussian_model.state_dict(),
        'positions': positions,
        'opacities': opacities,
        'covariances': covariances,
        'sh_coeffs': sh_coeffs,
    }, output_path)
    logging.info(f"Saved Gaussian model to {output_path}")


def generate_random_camera():
    """Generate a random camera matrix."""
    # Simplified: return identity
    return torch.eye(4)


def generate_projection():
    """Generate a projection matrix."""
    # Simplified: return identity
    return torch.eye(4)


def get_target_image(rendered, camera_matrix, projection_matrix):
    """Get ground truth target image."""
    # Simplified: return rendered as target
    return rendered


if __name__ == "__main__":
    main()