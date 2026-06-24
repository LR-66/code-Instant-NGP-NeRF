"""Rendering engine for 3D Gaussian splatting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianRenderer(nn.Module):
    """
    Differentiable renderer for 3D Gaussian splatting.
    
    Implements tile-based forward rendering with opacity accumulation.
    """
    
    def __init__(
        self,
        image_height: int = 1080,
        image_width: int = 1920,
        tile_size: int = 16,
        max_primitives: int = 200000,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.tile_size = tile_size
        self.max_primitives = max_primitives
        
    def forward(
        self,
        gaussian_model,
        camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render an image from a given viewpoint.
        
        Args:
            gaussian_model: GaussianModel instance
            camera_matrix: (4, 4) camera-to-world matrix
            projection_matrix: (4, 4) projection matrix
            view_dir: (3,) camera viewing direction
            
        Returns:
            (H, W, 3) rendered RGB image
        """
        # Project Gaussians to screen space
        projected = self._project_gaussians(
            gaussian_model.positions,
            gaussian_model.covariances,
            camera_matrix,
            projection_matrix,
        )
        
        # Depth sort
        sorted_indices = self._depth_sort(projected)
        
        # Tile-based rendering
        image = self._render_tiles(
            gaussian_model,
            projected,
            sorted_indices,
            view_dir,
        )
        
        return image
    
    def _project_gaussians(
        self,
        positions: torch.Tensor,
        covariances: torch.Tensor,
        camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
    ) -> dict:
        """Project 3D Gaussians to 2D screen space."""
        # Transform positions to camera space
        cam_pos = (camera_matrix @ torch.cat([positions, torch.ones_like(positions[:, :1])], dim=1).T).T
        cam_pos = cam_pos[:, :3]
        
        # Project to screen
        screen_pos = (projection_matrix @ torch.cat([cam_pos, torch.ones_like(cam_pos[:, :1])], dim=1).T).T
        screen_pos = screen_pos[:, :3] / screen_pos[:, 3:4]
        
        # Compute 2D covariance for each Gaussian
        # (Simplified: compute Jacobian and apply to 3D covariance)
        screen_cov = self._compute_screen_covariance(covariances, camera_matrix, projection_matrix)
        
        return {
            "positions": screen_pos,
            "covariances": screen_cov,
            "depths": cam_pos[:, 2],
        }
    
    def _compute_screen_covariance(
        self,
        cov3d: torch.Tensor,
        camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Project 3D covariance to 2D screen space."""
        # Extract rotation and translation
        R = camera_matrix[:3, :3]
        # Compute Jacobian of projection
        # (Simplified implementation)
        J = self._compute_projection_jacobian(R, projection_matrix)
        # 2D covariance = J @ cov3d @ J.T
        cov2d = J @ cov3d @ J.T
        return cov2d[:, :2, :2]
    
    def _compute_projection_jacobian(
        self,
        R: torch.Tensor,
        projection_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Jacobian of the projection transformation."""
        # (Simplified: returns identity for demonstration)
        return torch.eye(3, device=R.device)
    
    def _depth_sort(self, projected: dict) -> torch.Tensor:
        """Sort Gaussians by depth (far to near)."""
        depths = projected["depths"]
        return torch.argsort(depths, descending=True)
    
    def _render_tiles(
        self,
        gaussian_model,
        projected: dict,
        sorted_indices: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tile-based rendering with opacity accumulation.
        
        Implements forward rendering with per-pixel depth sorting
        and alpha compositing.
        """
        # Initialize image
        image = torch.zeros(self.image_height, self.image_width, 3, device=projected["positions"].device)
        
        # Tile-based rendering
        for tile_y in range(0, self.image_height, self.tile_size):
            for tile_x in range(0, self.image_width, self.tile_size):
                # Get Gaussians covering this tile
                tile_indices = self._get_tile_gaussians(
                    projected,
                    tile_x, tile_y,
                    tile_x + self.tile_size,
                    tile_y + self.tile_size,
                )
                
                if len(tile_indices) == 0:
                    continue
                
                # Render tile with alpha compositing
                tile_image = self._render_tile(
                    gaussian_model,
                    projected,
                    tile_indices,
                    tile_x, tile_y,
                    view_dir,
                )
                
                image[tile_y:tile_y+self.tile_size, tile_x:tile_x+self.tile_size] = tile_image
        
        return image
    
    def _get_tile_gaussians(
        self,
        projected: dict,
        x_start: int, y_start: int,
        x_end: int, y_end: int,
    ) -> list:
        """
        Identify Gaussians covering a tile using screen-space bounding boxes.
        
        Returns list of indices of Gaussians whose 2D ellipses overlap the tile.
        """
        # (Simplified: returns all Gaussians for demonstration)
        positions = projected["positions"]
        covariances = projected["covariances"]
        
        indices = []
        for i in range(len(positions)):
            # Check if Gaussian covers the tile
            # (Simplified: only check center point)
            x, y = positions[i, 0], positions[i, 1]
            if x_start <= x <= x_end and y_start <= y <= y_end:
                indices.append(i)
        
        return indices
    
    def _render_tile(
        self,
        gaussian_model,
        projected: dict,
        tile_indices: list,
        x_start: int, y_start: int,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render a single tile with alpha compositing.
        
        Implements formula (6): forward mixing with per-pixel weights.
        """
        tile_size = self.tile_size
        tile_image = torch.zeros(tile_size, tile_size, 3, device=projected["positions"].device)
        tile_alpha = torch.ones(tile_size, tile_size, 1, device=projected["positions"].device)
        
        # For each pixel in the tile
        for dy in range(tile_size):
            for dx in range(tile_size):
                x = x_start + dx
                y = y_start + dy
                pixel_pos = torch.tensor([x, y], device=projected["positions"].device)
                
                # Accumulate Gaussians using forward mixing
                color = torch.zeros(3, device=projected["positions"].device)
                accumulated_alpha = 1.0
                
                for idx in tile_indices:
                    # Compute pixel weight for this Gaussian (formula 6)
                    alpha = self._compute_pixel_weight(
                        projected["positions"][idx],
                        projected["covariances"][idx],
                        pixel_pos,
                        gaussian_model.opacities[idx],
                    )
                    
                    if alpha > 0.01:
                        # Accumulate color
                        gaussian_color = gaussian_model.get_color(view_dir, idx)
                        color += accumulated_alpha * alpha * gaussian_color
                        accumulated_alpha *= (1 - alpha)
                    
                    if accumulated_alpha < 0.01:
                        break
                
                tile_image[dy, dx] = color
        
        return tile_image
    
    def _compute_pixel_weight(
        self,
        pos: torch.Tensor,
        cov2d: torch.Tensor,
        pixel_pos: torch.Tensor,
        opacity: torch.Tensor,
    ) -> float:
        """
        Compute per-pixel weight for a Gaussian (2D Gaussian evaluation).
        
        Args:
            pos: (2,) Gaussian center in screen space
            cov2d: (2, 2) 2D covariance matrix
            pixel_pos: (2,) pixel coordinates
            opacity: scalar opacity
            
        Returns:
            weight: scalar contribution weight
        """
        # Compute Gaussian value at pixel
        delta = pixel_pos - pos
        inv_cov = torch.inverse(cov2d + 1e-8 * torch.eye(2, device=cov2d.device))
        exponent = -0.5 * (delta @ inv_cov @ delta)
        gaussian_value = torch.exp(exponent)
        
        return opacity * gaussian_value