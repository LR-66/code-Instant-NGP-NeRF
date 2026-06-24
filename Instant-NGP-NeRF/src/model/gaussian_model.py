"""3D Gaussian splatting model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianModel(nn.Module):
    """
    Differentiable 3D Gaussian splatting model.
    
    Represents a scene as a set of anisotropic 3D Gaussians with
    position, covariance, opacity, and spherical harmonic coefficients.
    """
    
    def __init__(
        self,
        max_primitives: int = 200000,
        sh_degree: int = 3,
        learn_lr_pos: float = 1.6e-4,
        learn_lr_cov: float = 5.0e-3,
        learn_lr_sh: float = 2.5e-3,
    ):
        super().__init__()
        self.max_primitives = max_primitives
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        
        # Parameters will be initialized during distillation
        self.register_buffer("_positions", torch.empty(0, 3))
        self.register_buffer("_opacities", torch.empty(0, 1))
        self.register_buffer("_covariances", torch.empty(0, 6))  # 6 DOF
        self.register_buffer("_sh_coeffs", torch.empty(0, self.num_sh_coeffs * 3))
        
        # Optimizer parameters
        self.learn_lr_pos = learn_lr_pos
        self.learn_lr_cov = learn_lr_cov
        self.learn_lr_sh = learn_lr_sh
        
    @property
    def positions(self):
        return self._positions
    
    @positions.setter
    def positions(self, value):
        self._positions = value
        
    @property
    def opacities(self):
        return torch.sigmoid(self._opacities)
    
    @property
    def covariances(self):
        # Convert from 6-DOF representation to 3x3 matrix
        return self._covariances
    
    def get_covariance_matrix(self, idx: int) -> torch.Tensor:
        """Get full 3x3 covariance matrix for a Gaussian."""
        cov_flat = self._covariances[idx]
        # Extract rotation (quaternion) and scale
        q = cov_flat[:4]
        s = cov_flat[4:]
        # Convert to 3x3 matrix
        R = self._quaternion_to_matrix(q)
        S = torch.diag(s)
        return R @ S @ S @ R.T
        
    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], device=q.device)
    
    def get_color(self, view_dir: torch.Tensor, idx: int) -> torch.Tensor:
        """Compute view-dependent color using spherical harmonics."""
        sh_coeffs = self._sh_coeffs[idx].reshape(self.num_sh_coeffs, 3)
        # Evaluate spherical harmonics at view direction
        sh_basis = self._compute_sh_basis(view_dir)
        color = (sh_basis @ sh_coeffs).clamp(0, 1)
        return color
    
    def _compute_sh_basis(self, view_dir: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical harmonic basis functions up to sh_degree.
        
        Args:
            view_dir: (3,) viewing direction
            
        Returns:
            (num_sh_coeffs,) basis function values
        """
        x, y, z = view_dir
        basis = []
        
        # Degree 0
        basis.append(1.0)
        
        if self.sh_degree >= 1:
            # Degree 1
            basis.extend([x, y, z])
            
        if self.sh_degree >= 2:
            # Degree 2
            basis.extend([
                x*y, x*z, y*z,
                x*x - y*y,
                3*z*z - 1,
            ])
            
        if self.sh_degree >= 3:
            # Degree 3
            basis.extend([
                x*(y*y - z*z),
                y*(z*z - x*x),
                z*(x*x - y*y),
                x*(x*x - 3*y*y),
                y*(3*x*x - y*y),
                z*(3*x*x - z*z),
                x*(3*y*y - z*z),
                y*(3*z*z - x*x),
                z*(3*x*x - y*y),
                (x*x + y*y + z*z)**1.5,
            ])
            
        return torch.tensor(basis, device=view_dir.device)
    
    def transform(self, quaternion: torch.Tensor, translation: torch.Tensor):
        """
        Apply rigid transformation to all Gaussians.
        
        Args:
            quaternion: (4,) rotation quaternion
            translation: (3,) translation vector
        """
        R = self._quaternion_to_matrix(quaternion)
        self._positions = (R @ self._positions.T).T + translation
        
        # Update covariance matrices
        for i in range(len(self._covariances)):
            cov = self.get_covariance_matrix(i)
            cov_new = R @ cov @ R.T
            # Convert back to 6-DOF representation
            self._covariances[i] = self._matrix_to_covariance(cov_new)
    
    def _matrix_to_covariance(self, cov: torch.Tensor) -> torch.Tensor:
        """Convert 3x3 covariance matrix to 6-DOF representation."""
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(cov)
        # Extract quaternion from rotation matrix
        q = self._matrix_to_quaternion(eigvecs)
        s = torch.sqrt(eigvals.clamp(min=1e-8))
        return torch.cat([q, s])
    
    def _matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion."""
        trace = R.trace()
        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            # Handle negative trace
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