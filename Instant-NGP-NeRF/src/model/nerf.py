"""NeRF model with multi-resolution hash encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hash_encoder import HashEncoder, AdaptiveHashEncoder


class NeRF(nn.Module):
    """
    NeRF model with multi-resolution hash encoding.
    
    Combines hash encoding with a lightweight MLP to predict density and color.
    """
    
    def __init__(
        self,
        num_levels: int = 16,
        min_resolution: int = 16,
        max_resolution: int = 2048,
        feature_dim: int = 2,
        hash_table_size: int = 2**19,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        adaptive: bool = False,
    ):
        super().__init__()
        
        if adaptive:
            self.hash_encoder = AdaptiveHashEncoder(
                num_levels=num_levels,
                min_resolution=min_resolution,
                max_resolution=max_resolution,
                feature_dim=feature_dim,
                hash_table_size=hash_table_size,
            )
        else:
            self.hash_encoder = HashEncoder(
                num_levels=num_levels,
                min_resolution=min_resolution,
                max_resolution=max_resolution,
                feature_dim=feature_dim,
                hash_table_size=hash_table_size,
            )
        
        input_dim = num_levels * feature_dim
        
        # Density branch
        self.density_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive density
        )
        
        # Color branch (view-dependent)
        self.color_mlp = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # +3 for view direction
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, view_dir: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: (N, 3) spatial coordinates
            view_dir: (N, 3) viewing direction
            
        Returns:
            density: (N, 1) volume density
            color: (N, 3) RGB color
        """
        features = self.hash_encoder(x)
        density = self.density_mlp(features)
        
        # Concatenate features with view direction for color
        color_input = torch.cat([features, view_dir], dim=-1)
        color = self.color_mlp(color_input)
        
        return density, color
    
    def get_density(self, x: torch.Tensor) -> torch.Tensor:
        """Get density only (for sampling and distillation)."""
        features = self.hash_encoder(x)
        return self.density_mlp(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoded features only (for visualization/analysis)."""
        return self.hash_encoder(x)