"""
Training script for U-Net underwater image enhancement model.

This script trains a U-Net model for underwater image enhancement using
paired degraded and clean underwater images.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import the U-Net model from our application
import sys
sys.path.append('.')
from app.models import UNet


class UnderwaterImageDataset(Dataset):
    """
    Dataset for underwater image enhancement.
    
    Assumes paired images:
    - Input: degraded underwater images
    - Target: enhanced/clean underwater images
    """
    
    def __init__(self, input_dir, target_dir=None, transform=None, image_size=(256, 256)):
        """
        Args:
            input_dir: Directory containing degraded underwater images
            target_dir: Directory containing clean/enhanced images (if available)
            transform: Optional transforms to apply
            image_size: Size to resize images to
        """
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir) if target_dir else None
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = sorted([
            f for f in self.input_dir.glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        print(f"Found {len(self.image_files)} images in {input_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load input image
        input_path = self.image_files[idx]
        input_img = Image.open(input_path).convert('RGB')
        
        # Load target image if available
        if self.target_dir:
            target_path = self.target_dir / input_path.name
            if target_path.exists():
                target_img = Image.open(target_path).convert('RGB')
            else:
                # If no target, use input as target (self-supervised)
                target_img = input_img.copy()
        else:
            # Self-supervised: enhance the same image
            target_img = input_img.copy()
        
        # Resize
        input_img = input_img.resize(self.image_size)
        target_img = target_img.resize(self.image_size)
        
        # Apply transforms
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        else:
            # Default transform
            to_tensor = transforms.ToTensor()
            input_img = to_tensor(input_img)
            target_img = to_tensor(target_img)
        
        return input_img, target_img


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    Helps preserve content while enhancing images.
    """
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use pretrained VGG for perceptual loss
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
            for param in self.features.parameters():
                param.requires_grad = False
        except:
            self.features = None
    
    def forward(self, pred, target):
        if self.features is None:
            return 0
        
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        return nn.functional.mse_loss(pred_features, target_features)


class CombinedLoss(nn.Module):
    """
    Combined loss function for image enhancement:
    - MSE Loss: Pixel-wise similarity
    - Perceptual Loss: Content preservation
    - SSIM Loss: Structural similarity
    """
    
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, ssim_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        # MSE Loss
        mse = self.mse_loss(pred, target)
        
        # Perceptual Loss
        perceptual = self.perceptual_loss(pred, target)
        
        # Total loss
        total_loss = (
            self.mse_weight * mse +
            self.perceptual_weight * perceptual
        )
        
        return total_loss, {'mse': mse.item(), 'perceptual': perceptual.item()}


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_perceptual = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        running_mse += loss_dict['mse']
        running_perceptual += loss_dict['perceptual']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'mse': running_mse / (batch_idx + 1),
            'perceptual': running_perceptual / (batch_idx + 1)
        })
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss, _ = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(dataloader)


def save_sample_images(model, dataloader, device, save_dir, epoch):
    """Save sample enhanced images."""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Save first 4 samples
        num_samples = min(4, inputs.size(0))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Input image
            input_img = inputs[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            # Enhanced image
            output_img = outputs[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 1].imshow(output_img)
            axes[i, 1].set_title('Enhanced')
            axes[i, 1].axis('off')
            
            # Target image
            target_img = targets[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 2].imshow(target_img)
            axes[i, 2].set_title('Target')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'samples_epoch_{epoch}.png')
        plt.close()


def train_unet(
    train_dir,
    val_dir=None,
    target_train_dir=None,
    target_val_dir=None,
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-4,
    image_size=(256, 256),
    save_dir='training_output',
    checkpoint_interval=10
):
    """
    Train U-Net model for underwater image enhancement.
    
    Args:
        train_dir: Directory with training images
        val_dir: Directory with validation images (optional)
        target_train_dir: Directory with target training images (optional)
        target_val_dir: Directory with target validation images (optional)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        image_size: Input image size
        save_dir: Directory to save outputs
        checkpoint_interval: Save checkpoint every N epochs
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = UnderwaterImageDataset(
        train_dir,
        target_train_dir,
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Validation dataset
    if val_dir:
        val_dataset = UnderwaterImageDataset(
            val_dir,
            target_val_dir,
            image_size=image_size
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Initialize model
    print("\nInitializing model...")
    model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(mse_weight=1.0, perceptual_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_dataset)}")
    print("=" * 60)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_dir / 'best_model.pth')
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Save sample images
        if epoch % checkpoint_interval == 0:
            save_sample_images(model, train_loader, device, save_dir / 'samples', epoch)
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    final_path = save_dir / 'enhancer_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training complete! Final model saved to: {final_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    if val_loader:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_history.png')
    plt.close()
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train U-Net for underwater image enhancement')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory with training images')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Directory with validation images')
    parser.add_argument('--target_train_dir', type=str, default=None,
                        help='Directory with target training images')
    parser.add_argument('--target_val_dir', type=str, default=None,
                        help='Directory with target validation images')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (will be resized to NxN)')
    parser.add_argument('--save_dir', type=str, default='training_output',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Train model
    train_unet(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        target_train_dir=args.target_train_dir,
        target_val_dir=args.target_val_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        image_size=(args.image_size, args.image_size),
        save_dir=args.save_dir
    )
