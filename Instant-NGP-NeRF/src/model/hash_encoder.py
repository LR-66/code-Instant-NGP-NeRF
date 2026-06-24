"""Multi-resolution hash encoding for Instant-NGP."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HashEncoder(nn.Module):
    """
    Multi-resolution hash encoding as described in Instant-NGP.
    
    Encodes 3D coordinates into features using a multi-resolution hash table.
    """
    
    def __init__(
        self,
        num_levels: int = 16,
        min_resolution: int = 16,
        max_resolution: int = 2048,
        feature_dim: int = 2,
        hash_table_size: int = 2**19,
        log2_hashmap_size: int = 19,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.feature_dim = feature_dim
        self.hash_table_size = hash_table_size
        
        # Exponentially increasing resolution per level
        self.resolutions = torch.exp(
            torch.linspace(np.log(min_resolution), np.log(max_resolution), num_levels)
        ).int()
        
        # Hash table for each level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(
                torch.randn(hash_table_size, feature_dim) * 0.01,
                requires_grad=True
            )
            for _ in range(num_levels)
        ])
        
        # Primitive constants for hash function
        self.primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
        
    def _hash_function(self, indices: torch.Tensor) -> torch.Tensor:
        """Apply the hash function to grid indices."""
        # indices: (N, 3)
        h = indices[:, 0] ^ (indices[:, 1] * self.primes[1]) ^ (indices[:, 2] * self.primes[2])
        return h % self.hash_table_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D coordinates.
        
        Args:
            x: (N, 3) coordinates in [-1, 1]^3
            
        Returns:
            (N, num_levels * feature_dim) encoded features
        """
        N = x.shape[0]
        features = []
        
        for level_idx in range(self.num_levels):
            res = self.resolutions[level_idx].item()
            
            # Scale coordinates to grid resolution
            grid_coords = (x + 1.0) * 0.5 * (res - 1)
            
            # Get corner indices
            idx_low = grid_coords.floor().long()
            idx_high = idx_low + 1
            
            # Clamp to [0, res-1]
            idx_low = torch.clamp(idx_low, 0, res - 1)
            idx_high = torch.clamp(idx_high, 0, res - 1)
            
            # Trilinear interpolation weights
            frac = grid_coords - grid_coords.floor()
            
            # Hash indices for 8 corners
            corners = torch.cat([
                idx_low[:, 0:1], idx_low[:, 1:2], idx_low[:, 2:3],
                idx_high[:, 0:1], idx_low[:, 1:2], idx_low[:, 2:3],
                idx_low[:, 0:1], idx_high[:, 1:2], idx_low[:, 2:3],
                idx_high[:, 0:1], idx_high[:, 1:2], idx_low[:, 2:3],
                idx_low[:, 0:1], idx_low[:, 1:2], idx_high[:, 2:3],
                idx_high[:, 0:1], idx_low[:, 1:2], idx_high[:, 2:3],
                idx_low[:, 0:1], idx_high[:, 1:2], idx_high[:, 2:3],
                idx_high[:, 0:1], idx_high[:, 1:2], idx_high[:, 2:3],
            ], dim=1).reshape(N, 8, 3)
            
            # Hash each corner
            hash_indices = self._hash_function(corners.reshape(-1, 3)).reshape(N, 8)
            
            # Retrieve features from hash table
            hash_table = self.hash_tables[level_idx]
            corner_features = hash_table[hash_indices]  # (N, 8, feature_dim)
            
            # Trilinear interpolation weights
            w = torch.stack([
                (1 - frac[:, 0]) * (1 - frac[:, 1]) * (1 - frac[:, 2]),
                (frac[:, 0]) * (1 - frac[:, 1]) * (1 - frac[:, 2]),
                (1 - frac[:, 0]) * (frac[:, 1]) * (1 - frac[:, 2]),
                (frac[:, 0]) * (frac[:, 1]) * (1 - frac[:, 2]),
                (1 - frac[:, 0]) * (1 - frac[:, 1]) * (frac[:, 2]),
                (frac[:, 0]) * (1 - frac[:, 1]) * (frac[:, 2]),
                (1 - frac[:, 0]) * (frac[:, 1]) * (frac[:, 2]),
                (frac[:, 0]) * (frac[:, 1]) * (frac[:, 2]),
            ], dim=1).unsqueeze(-1)  # (N, 8, 1)
            
            # Weighted sum
            level_feature = (corner_features * w).sum(dim=1)  # (N, feature_dim)
            features.append(level_feature)
        
        return torch.cat(features, dim=-1)  # (N, num_levels * feature_dim)


class AdaptiveHashEncoder(HashEncoder):
    """
    Adaptive hash encoder with geometry-aware resolution allocation.
    
    Extends the base hash encoder with adaptive density control that
    dynamically upgrades hash resolution in high-gradient regions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometric_complexity = None
        self.beta = 0.9
        
    def update_complexity(self, density_gradients: torch.Tensor):
        """Update geometric complexity estimates per spatial location."""
        grad_norm = torch.norm(density_gradients, dim=-1)
        if self.geometric_complexity is None:
            self.geometric_complexity = grad_norm.clone()
        else:
            self.geometric_complexity = self.beta * self.geometric_complexity + (1 - self.beta) * grad_norm
            
    def get_adaptive_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute resolution allocation weights based on local complexity.
        
        Returns weights that prioritize higher resolution in complex regions.
        """
        if self.geometric_complexity is None:
            return torch.ones(x.shape[0], self.num_levels, device=x.device) / self.num_levels
        
        # Interpolate complexity to query points
        # (Simplified: use nearest neighbor interpolation)
        weights = self.geometric_complexity / (self.geometric_complexity.sum() + 1e-8)
        return weights.unsqueeze(-1).expand(-1, self.num_levels)