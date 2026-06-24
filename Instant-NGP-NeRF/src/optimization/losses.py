"""Loss functions for training and distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VolumeRenderingLoss(nn.Module):
    """Loss for NeRF training."""
    
    def __init__(self, l1_weight: float = 1.0, depth_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.depth_weight = depth_weight
        
    def forward(self, rendered, target, depth=None, target_depth=None):
        l1_loss = F.l1_loss(rendered, target)
        
        if depth is not None and target_depth is not None:
            depth_loss = F.mse_loss(depth, target_depth)
            total = self.l1_weight * l1_loss + self.depth_weight * depth_loss
        else:
            total = l1_loss
            
        return total


class GaussianDistillationLoss(nn.Module):
    """Loss for Gaussian distillation."""
    
    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # Pre-trained VGG for perceptual loss
        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Normalization for VGG input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, rendered, target):
        l1_loss = F.l1_loss(rendered, target)
        
        # Perceptual loss
        rendered_norm = (rendered - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features from multiple layers
        rendered_features = self._get_features(rendered_norm)
        target_features = self._get_features(target_norm)
        
        perceptual_loss = 0.0
        for rf, tf in zip(rendered_features, target_features):
            perceptual_loss += F.mse_loss(rf, tf)
        
        total = self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss
        return total
    
    def _get_features(self, x):
        """Extract features from VGG layers."""
        features = []
        layers = [4, 9, 18, 27, 36]  # Conv layers indices
        
        with torch.no_grad():
            for i, layer in enumerate(self.vgg.children()):
                x = layer(x)
                if i in layers:
                    features.append(x)
        return features


class MultiViewConsistencyLoss(nn.Module):
    """
    Multi-view geometric consistency loss.
    
    Implements equations (12) and (13): photometric and depth consistency.
    """
    
    def __init__(self, photo_weight: float = 1.0, depth_weight: float = 0.5):
        super().__init__()
        self.photo_weight = photo_weight
        self.depth_weight = depth_weight
        
    def forward(self, rendered_a, target_a, rendered_b, target_b,
                depth_a, depth_b, proj_a_to_b):
        """
        Compute cross-view consistency loss.
        
        Args:
            rendered_a: Image from view A
            target_a: Target from view A
            rendered_b: Image from view B
            target_b: Target from view B
            depth_a: Depth from view A
            depth_b: Depth from view B
            proj_a_to_b: Projection mapping from A to B
        """
        # Photometric consistency (equation 12)
        reprojected_b = proj_a_to_b(rendered_a, depth_a)
        photo_loss = F.l1_loss(reprojected_b, target_b)
        
        # Depth consistency (equation 13)
        reprojected_depth = proj_a_to_b(depth_a)
        depth_loss = F.l1_loss(reprojected_depth, depth_b)
        
        total = self.photo_weight * photo_loss + self.depth_weight * depth_loss
        return total


class RigidBodyConstraintLoss(nn.Module):
    """
    Rigid-body constraint for grouped Gaussian primitives.
    
    Implements equation (11): maintains relative distances within components.
    """
    
    def __init__(self, distance_threshold: float = 0.05):
        super().__init__()
        self.distance_threshold = distance_threshold
        
    def forward(self, positions, initial_positions, neighbor_pairs):
        """
        Compute rigid-body constraint loss.
        
        Args:
            positions: Current Gaussian positions
            initial_positions: Initial Gaussian positions
            neighbor_pairs: List of (i, j, initial_distance) tuples
        """
        loss = 0.0
        
        for i, j, initial_dist in neighbor_pairs:
            current_dist = torch.norm(positions[i] - positions[j])
            dist_error = (current_dist - initial_dist) ** 2
            
            # Only penalize large deviations
            if dist_error > self.distance_threshold ** 2:
                loss += dist_error
        
        return loss


class TotalVariationLoss(nn.Module):
    """Total variation regularization for smoothness."""
    
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
        
    def forward(self, image):
        """
        Compute total variation loss.
        
        Implements equation (15): sum of horizontal and vertical gradients.
        """
        tv_h = torch.abs(image[:, 1:, :] - image[:, :-1, :]).sum()
        tv_w = torch.abs(image[:, :, 1:] - image[:, :, :-1]).sum()
        return self.weight * (tv_h + tv_w)