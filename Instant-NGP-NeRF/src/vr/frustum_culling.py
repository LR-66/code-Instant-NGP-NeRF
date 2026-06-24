"""View-frustum culling for VR interaction."""

import torch
import numpy as np


class FrustumCuller:
    """
    Efficient view-frustum culling using bounding sphere hierarchy.
    
    Implements equation (8): visibility test based on clipping plane distances.
    """
    
    def __init__(self, b_tree_depth: int = 8, leaf_size: int = 32):
        self.b_tree_depth = b_tree_depth
        self.leaf_size = leaf_size
        self.bounding_radius_scale = 1.2
        
    def build_hierarchy(self, positions, covariance_matrices):
        """
        Build bounding sphere hierarchy (B-tree).
        
        Args:
            positions: (N, 3) Gaussian centers
            covariance_matrices: (N, 3, 3) Gaussian covariances
            
        Returns:
            hierarchy: List of node indices and bounding sphere radii
        """
        # Compute bounding sphere radius for each Gaussian
        # Definition: longest eigen-axis length of covariance * 1.2
        radii = []
        for cov in covariance_matrices:
            eigvals = torch.linalg.eigvalsh(cov)
            radius = torch.sqrt(eigvals.max()) * self.bounding_radius_scale
            radii.append(radius)
        radii = torch.tensor(radii)
        
        # Build hierarchical structure (simplified implementation)
        # Returns indices organized for efficient traversal
        
        # For this implementation, we keep a simple flat list
        hierarchy = {
            'positions': positions,
            'radii': radii,
            'child_indices': torch.arange(len(positions)),
        }
        
        return hierarchy
    
    def cull(self, hierarchy, frustum_planes):
        """
        Cull Gaussians outside the view frustum.
        
        Implements equation (8):
        V_i = ∏_{k=1}^{6} I(n_k^T μ_i + d_k + r_i > 0)
        
        Args:
            hierarchy: B-tree hierarchy
            frustum_planes: List of 6 planes (n_k, d_k)
            
        Returns:
            visible_indices: Boolean mask of visible Gaussians
        """
        positions = hierarchy['positions']
        radii = hierarchy['radii']
        
        visible = torch.ones(len(positions), dtype=torch.bool, device=positions.device)
        
        for n, d in frustum_planes:
            # Compute signed distances for all Gaussians
            distances = torch.matmul(positions, n) + d + radii
            visible = visible & (distances > 0)
        
        return visible
    
    def compute_frustum_planes(self, view_matrix, projection_matrix):
        """
        Compute 6 clipping planes from view and projection matrices.
        
        Returns:
            List of (normal, offset) tuples for 6 planes.
        """
        # Combine view and projection
        M = projection_matrix @ view_matrix
        
        # Extract frustum planes
        planes = []
        
        # Left plane
        n = M[3, :3] + M[0, :3]
        d = M[3, 3] + M[0, 3]
        n = n / torch.norm(n)
        planes.append((n, d / torch.norm(n)))
        
        # Right plane
        n = M[3, :3] - M[0, :3]
        d = M[3, 3] - M[0, 3]
        n = n / torch.norm(n)
        planes.append((n, d / torch.norm(n)))
        
        # Bottom plane
        n = M[3, :3] + M[1, :3]
        d = M[3, 3] + M[1, 3]
        n = n / torch.norm(n)
        planes.append((n, d / torch.norm(n)))
        
        # Top plane
        n = M[3, :3] - M[1, :3]
        d = M[3, 3] - M[1, 3]
        n = n / torch.norm(n)
        planes.append((n, d / torch.norm(n)))
        
        # Near plane
        n = M[3, :3] + M[2, :3]
        d = M[3, 3] + M[2, 3]
        n = n / torch.norm(n)
        planes.append((n, d / torch.norm(n)))
        
        # Far plane
        n = M[3, :3] - M[2, :3]
        d = M[3, 3] - M[2, 3]
        n = n / torch.norm(n)
        planes.append((n, d / torch.norm(n)))
        
        return planes