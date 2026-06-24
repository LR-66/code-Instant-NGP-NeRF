"""Dynamic attribute update for interactive VR components."""

import torch
import numpy as np


class DynamicUpdater:
    """
    Handles dynamic updates to Gaussian attributes during VR interaction.
    
    Implements:
    - Component-level active-passive separation
    - Rigid-body transformations (equations 9, 10)
    - Rigid-body constraint monitoring (equation 11)
    """
    
    def __init__(self, rigid_constraint_weight: float = 0.01):
        self.rigid_constraint_weight = rigid_constraint_weight
        self.component_hierarchy = {}
        self.component_groups = {}
        self.initial_distances = {}
        
    def register_component(self, component_id: str, gaussian_indices: torch.Tensor):
        """
        Register a teaching component with its Gaussian indices.
        
        Args:
            component_id: Unique component identifier
            gaussian_indices: Indices of Gaussians belonging to this component
        """
        self.component_groups[component_id] = gaussian_indices
    
    def compute_initial_distances(self, positions, component_id: str):
        """
        Compute initial pair distances for rigid-body constraint.
        
        Implements equation (11): monitors distance variations.
        """
        indices = self.component_groups[component_id]
        comp_positions = positions[indices]
        
        distances = []
        pairs = []
        
        for i in range(len(comp_positions)):
            for j in range(i + 1, len(comp_positions)):
                dist = torch.norm(comp_positions[i] - comp_positions[j])
                distances.append(dist)
                pairs.append((i, j))
        
        self.initial_distances[component_id] = (torch.tensor(distances), pairs)
        return self.initial_distances[component_id]
    
    def transform_component(
        self,
        positions,
        covariances,
        component_id: str,
        quaternion: torch.Tensor,
        translation: torch.Tensor,
    ):
        """
        Apply rigid transformation to a component.
        
        Implements equation (9): μ_i' = R(q) μ_i + t
        Implements equation (10): Σ_i' = R(q) Σ_i R^T(q)
        
        Args:
            positions: (N, 3) all Gaussian positions
            covariances: (N, 6) all covariance parameters (6-DOF)
            component_id: Component to transform
            quaternion: (4,) rotation quaternion
            translation: (3,) translation vector
            
        Returns:
            Updated positions and covariances
        """
        indices = self.component_groups[component_id]
        
        # Compute rotation matrix from quaternion
        R = self._quaternion_to_matrix(quaternion)
        
        # Update positions (equation 9)
        comp_positions = positions[indices]
        positions[indices] = (R @ comp_positions.T).T + translation
        
        # Update covariances (equation 10)
        for i, idx in enumerate(indices):
            cov_flat = covariances[idx]
            cov_matrix = self._flat_to_matrix(cov_flat)
            cov_matrix_new = R @ cov_matrix @ R.T
            covariances[idx] = self._matrix_to_flat(cov_matrix_new)
        
        return positions, covariances
    
    def check_rigid_constraint(self, positions, component_id: str) -> float:
        """
        Check rigid-body constraint and return violation loss.
        
        Implements equation (11):
        ℒ_rigid = Σ_{(i,j)∈N} (||μ_i' - μ_j'||₂ - d_ij)²
        
        Returns:
            Violation loss (0 if no violation)
        """
        if component_id not in self.initial_distances:
            return 0.0
        
        indices = self.component_groups[component_id]
        comp_positions = positions[indices]
        initial_dists, pairs = self.initial_distances[component_id]
        
        loss = 0.0
        threshold = 0.05  # Distance change threshold
        
        for k, (i, j) in enumerate(pairs):
            current_dist = torch.norm(comp_positions[i] - comp_positions[j])
            init_dist = initial_dists[k]
            dist_error = (current_dist - init_dist) ** 2
            
            # Only penalize large deviations
            if dist_error > threshold ** 2:
                loss += dist_error
        
        return loss * self.rigid_constraint_weight
    
    def _quaternion_to_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], device=q.device)
    
    def _flat_to_matrix(self, flat):
        """Convert 6-DOF to 3x3 matrix."""
        q = flat[:4]
        s = flat[4:].clamp(min=1e-6)
        R = self._quaternion_to_matrix(q)
        return R @ torch.diag(s) @ R.T
    
    def _matrix_to_flat(self, matrix):
        """Convert 3x3 matrix to 6-DOF."""
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        s = torch.sqrt(eigvals.clamp(min=1e-6))
        
        # Convert eigenvector matrix to quaternion
        q = self._matrix_to_quaternion(eigvecs)
        return torch.cat([q, s])
    
    def _matrix_to_quaternion(self, R):
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