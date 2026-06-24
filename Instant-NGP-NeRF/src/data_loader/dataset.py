"""Dataset loader for multi-view image sequences."""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class NeRFDataset(Dataset):
    """
    NeRF dataset loader for multi-view images with camera poses.
    
    Expects COLMAP-style transforms.json format.
    """
    
    def __init__(
        self,
        root_dir: str,
        scene_name: str,
        image_downscale: int = 1,
        split: str = 'train',
        train_ratio: float = 0.8,
    ):
        self.root_dir = os.path.join(root_dir, scene_name)
        self.image_downscale = image_downscale
        self.split = split
        self.train_ratio = train_ratio
        
        # Load camera transforms
        transforms_path = os.path.join(self.root_dir, 'transforms.json')
        with open(transforms_path, 'r') as f:
            self.transforms = json.load(f)
        
        self.frames = self.transforms['frames']
        
        # Split data
        n_frames = len(self.frames)
        n_train = int(n_frames * train_ratio)
        
        if split == 'train':
            self.frames = self.frames[:n_train]
        elif split == 'val':
            self.frames = self.frames[n_train:]
        elif split == 'test':
            self.frames = self.frames[n_train:]  # Use remaining for test
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Load image
        image_path = os.path.join(self.root_dir, frame['file_path'])
        image = Image.open(image_path).convert('RGB')
        
        if self.image_downscale > 1:
            w, h = image.size
            image = image.resize((w // self.image_downscale, h // self.image_downscale))
        
        image_tensor = self.transform(image)  # (3, H, W)
        
        # Load camera pose
        camera_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'camera_matrix': camera_matrix,
            'file_path': frame['file_path'],
            'idx': idx,
        }
    
    def get_rays(self, idx):
        """
        Generate rays for a given image.
        
        Returns:
            rays_o: (H*W, 3) ray origins
            rays_d: (H*W, 3) ray directions
        """
        data = self[idx]
        image = data['image']
        camera_matrix = data['camera_matrix']
        
        H, W = image.shape[1], image.shape[2]
        
        # Generate pixel coordinates
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H),
            torch.linspace(0, W-1, W),
            indexing='ij'
        )
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)
        
        # Transform to camera coordinates
        # (Simplified: assumes normalized device coordinates)
        rays_o = camera_matrix[:3, 3].expand(H*W, -1)
        rays_d = (camera_matrix[:3, :3] @ pixels.T).T
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        return rays_o, rays_d
    
    def get_validation_split(self):
        """Get the validation subset."""
        val_indices = range(int(len(self) * self.train_ratio), len(self))
        return [self[i] for i in val_indices]