"""Training script for NeRF with multi-resolution hash encoding."""

import os
import sys
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.nerf import NeRF
from models.renderer import GaussianRenderer
from data_loader.dataset import NeRFDataset
from optimization.losses import VolumeRenderingLoss
from optimization.scheduler import CosineAnnealingWarmupLR


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    setup_logging(config['logging']['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load dataset
    dataset = NeRFDataset(
        root_dir=config['data']['root_dir'],
        scene_name=config['data']['scene_name'],
        image_downscale=config['data']['image_downscale'],
        split='train',
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Initialize model
    model = NeRF(
        num_levels=config['model']['hash_encoder']['num_levels'],
        min_resolution=config['model']['hash_encoder']['min_resolution'],
        max_resolution=config['model']['hash_encoder']['max_resolution'],
        feature_dim=config['model']['hash_encoder']['feature_dim'],
        hash_table_size=config['model']['hash_encoder']['hash_table_size'],
        hidden_dim=config['model']['mlp']['hidden_dim'],
        num_hidden_layers=config['model']['mlp']['num_layers'],
        adaptive=True,
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr_base'])
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_steps=config['training'].get('warmup_steps', 1000),
        total_steps=config['training']['num_epochs'],
    )
    
    # Loss
    criterion = VolumeRenderingLoss(
        l1_weight=config['loss']['l1_weight'],
        depth_weight=config['loss']['depth_weight'],
    )
    
    # Training loop
    best_psnr = 0.0
    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        for batch in pbar:
            # Move to device
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            # (Simplified: sample points along rays and render)
            density, color = model(rays_o, rays_d)
            
            # Compute loss
            loss = criterion(density, color, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0],
            })
        
        # Validation
        if (epoch + 1) % config['logging']['log_interval'] == 0:
            val_psnr = validate(model, dataset, device)
            logging.info(f"Epoch {epoch+1}: val_psnr={val_psnr:.2f} dB")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                # Save checkpoint
                checkpoint_path = os.path.join(
                    config['logging']['checkpoint_dir'],
                    'nerf_best.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")


def validate(model, dataset, device):
    """Run validation on the validation set."""
    model.eval()
    psnr_total = 0.0
    count = 0
    
    # Use validation split
    val_dataset = dataset.get_validation_split()
    
    with torch.no_grad():
        for batch in val_dataset:
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            target = batch['target'].to(device)
            
            density, color = model(rays_o, rays_d)
            
            # Compute PSNR for this batch
            mse = torch.mean((color - target) ** 2)
            psnr = 10 * torch.log10(1.0 / mse)
            psnr_total += psnr.item()
            count += 1
    
    return psnr_total / count if count > 0 else 0.0


if __name__ == "__main__":
    main()