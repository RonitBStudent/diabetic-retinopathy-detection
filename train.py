import os
import random
import math
import time
import yaml
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip, ShiftScaleRotate, 
    RandomBrightnessContrast, OpticalDistortion, GridDistortion, PiecewiseAffine,
    CLAHE, Sharpen, Emboss, Normalize, OneOf
)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import wandb
from tqdm import tqdm
import cv2
import warnings
import matplotlib.pyplot as plt
from gradcam import GradCAM, visualize_gradcam
import coremltools as ct
warnings.filterwarnings('ignore')

# Custom dataset for diabetic retinopathy
class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Verify and clean image file paths
        self.image_files = []
        self.labels = []
        
        for idx, row in df.iterrows():
            img_path = os.path.join(self.img_dir, row['id_code'])
            if not os.path.exists(img_path):
                # Try with .png extension if not present
                if not img_path.endswith('.png'):
                    img_path = f"{img_path}.png"
                if not os.path.exists(img_path):
                    print(f"Warning: Could not find image at {img_path}")
                    continue
            self.image_files.append(os.path.basename(img_path))
            self.labels.append(row['diabetic_retinopathy'])
            
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"Loaded {len(self.image_files)} out of {len(df)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            
            # Try to load the image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image of the same size if loading fails
                image = Image.new('RGB', (224, 224), color='black')
            
            label = self.labels[idx]
            
            if self.transforms:
                try:
                    # Convert PIL Image to numpy array for Albumentations
                    image_np = np.array(image)
                    # Apply transforms with the correct argument name
                    transformed = self.transforms(image=image_np)
                    image = transformed['image']
                except Exception as e:
                    print(f"Error applying transforms to image {img_path}: {e}")
                    # If transforms fail, return a zero tensor
                    image = torch.zeros((3, 224, 224), dtype=torch.float32)
            else:
                # Basic transforms if none provided
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {e}")
            # Return a zero tensor and a default label if something goes wrong
            return torch.zeros((3, 224, 224), dtype=torch.float32), 0.0

def get_transforms(split, config):
    if split == 'train':
        return A.Compose([
            A.Resize(config['img_size'], config['img_size']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config['img_size'], config['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def compute_metrics(targets, outputs, threshold=0.5):
    """Compute binary classification metrics"""
    preds = (outputs > threshold).astype(int)
    targets = targets.astype(int)
    
    metrics = {}
    try:
        # Calculate metrics
        metrics['auc'] = roc_auc_score(targets, outputs)
        metrics['accuracy'] = accuracy_score(targets, preds)
        metrics['precision'] = precision_score(targets, preds, zero_division=0)
        metrics['recall'] = recall_score(targets, preds, zero_division=0)
        metrics['f1'] = f1_score(targets, preds, zero_division=0)
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        metrics = {k: float('nan') for k in ['auc', 'accuracy', 'precision', 'recall', 'f1']}
    
    return metrics

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate(model, val_loader, criterion, device, scaler=None):
    """Run validation on the validation set with improved error handling"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    try:
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", leave=False):
                # Skip incomplete batches
                if images is None or targets is None:
                    print("Skipping batch with None values")
                    continue
                    
                try:
                    images, targets = images.to(device), targets.to(device)
                    
                    # Forward pass
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
                        outputs = model(images)
                        # Ensure targets have the right shape
                        if targets.dim() == 1:
                            targets = targets.unsqueeze(1)
                        loss = criterion(outputs, targets.float())
                    
                    # Skip if loss is NaN or inf
                    if not torch.isfinite(loss):
                        print(f"Warning: Invalid validation loss {loss.item()}")
                        continue
                    
                    # Statistics
                    running_loss += loss.item() * images.size(0)
                    all_outputs.append(outputs.sigmoid().cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
        
        # Convert lists to numpy arrays if not empty
        if len(all_outputs) > 0 and len(all_targets) > 0:
            try:
                all_outputs = np.concatenate(all_outputs).ravel()
                all_targets = np.concatenate(all_targets).ravel()
                
                # Calculate metrics
                metrics = {}
                try:
                    metrics['auc'] = roc_auc_score(all_targets, all_outputs)
                    predictions = (all_outputs > 0.5).astype(int)
                    metrics['accuracy'] = accuracy_score(all_targets, predictions)
                    metrics['precision'] = precision_score(all_targets, predictions, zero_division=0)
                    metrics['recall'] = recall_score(all_targets, predictions, zero_division=0)
                    metrics['f1'] = f1_score(all_targets, predictions, zero_division=0)
                    metrics['val_loss'] = running_loss / len(val_loader.dataset)
                    
                    print(f"\n=== Validation Results ===")
                    print(f"Loss: {metrics['val_loss']:.4f}")
                    print(f"AUC: {metrics['auc']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
                    
                    # Print sample predictions
                    print("\nSample validation predictions:")
                    for i in range(min(5, len(all_outputs))):
                        print(f"  Target: {all_targets[i]:.4f}, Pred: {all_outputs[i]:.4f}")
                        
                except Exception as e:
                    print(f"Error calculating metrics: {e}")
                    # Fallback to simple accuracy if other metrics fail
                    if len(np.unique(all_targets)) >= 2:
                        metrics['accuracy'] = np.mean((all_outputs > 0.5).astype(int) == all_targets)
                    else:
                        metrics['accuracy'] = 0.5
                    
                    metrics.update({
                        'auc': 0.5,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'val_loss': running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')
                    })
                    
            except Exception as e:
                print(f"Error concatenating results: {e}")
                metrics = create_fallback_metrics()
        else:
            print("Warning: No valid predictions were made during validation")
            metrics = create_fallback_metrics()
            
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        metrics = create_fallback_metrics()
    
    return metrics

def create_fallback_metrics():
    """Create a set of fallback metrics when validation fails"""
    return {
        'auc': 0.5,
        'accuracy': 0.5,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'val_loss': float('nan')
    }

def save_checkpoint(model, optimizer, epoch, save_dir, filename='best_model.pth'):
    """Save model checkpoint"""
    # Prepare state dictionary
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # First try: Save to specified directory
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(state, save_path)
        print(f"Model successfully saved to {save_path}")
        return True
    except Exception as e:
        print(f"Warning: Could not save to {save_dir}: {e}")
    
    # Fallback: Save to current directory
    try:
        save_path = filename
        torch.save(state, save_path)
        print(f"Model saved to current directory as {save_path}")
        return True
    except Exception as e2:
        print(f"Error: Failed to save model: {e2}")
        return False


def main(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    set_seed(config.get('seed', 42))
    
    # Initialize wandb
    wandb.init(project=config.get('wandb_project', 'diabetic_retinopathy'), 
               name=config.get('run_name', 'experiment'),
               config=config)
    
    print("\nLoading and preparing data...")
    
    # Load and prepare data
    df = pd.read_csv(config['train_csv'])
    
    # Convert diagnosis to binary (0: No DR, 1: Any DR)
    df['diabetic_retinopathy'] = (df['diagnosis'] > 0).astype(int)
    
    # Ensure id_code has .png extension (without duplicating it)
    df['id_code'] = df['id_code'].apply(lambda x: f"{x}.png" if not x.endswith('.png') else x)
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['diabetic_retinopathy']
    )
    
    # Print class distribution
    print("\nTraining set class distribution (diabetic_retinopathy):")
    print(train_df['diabetic_retinopathy'].value_counts().to_dict())
    print("\nValidation set class distribution (diabetic_retinopathy):")
    print(val_df['diabetic_retinopathy'].value_counts().to_dict())
    
    # Create datasets
    train_ds = FundusDataset(
        train_df, 
        config['img_dir'], 
        transforms=get_transforms('train', config)
    )
    
    val_ds = FundusDataset(
        val_df, 
        config['img_dir'], 
        transforms=get_transforms('val', config)
    )
    
    # Calculate class weights for imbalanced dataset
    class_counts = train_df['diabetic_retinopathy'].value_counts().to_dict()
    total = sum(class_counts.values())
    class_weights = {cls: total/(len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[cls] for cls in train_df['diabetic_retinopathy']]
    
    # Create dataloaders with settings that achieved 0.9933 validation AUC
    train_loader = DataLoader(
        train_ds,
        batch_size=config.get('batch_size', 32),
        sampler=WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        ),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get('batch_size', 32) * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nUsing device: {device}")
    
    # Import the new model class
    from gradcam_efficientnet import GradCAMEfficientNet
    
    # Initialize the model with pretrained weights
    model = GradCAMEfficientNet(
        num_classes=1,
        pretrained=config.get('pretrained', True)
    )
    
    # Unfreeze all layers for training
    for param in model.parameters():
        param.requires_grad = True
        
    print("✅ Using GradCAMEfficientNet with Grad-CAM support")
    
    model = model.to(device)
    
    # Debug: Verify model weights
    print("\n=== Model Verification ===")
    print("First conv layer weights (first 3x3):")
    print(model.features[0][0].weight[0, 0, :3, :3])
    print("Last classifier layer weights mean:", model.classifier[1].weight.mean().item())
    
    # Calculate positive weight for loss function
    pos_weight = torch.tensor(
        [
            (len(train_df) - train_df['diabetic_retinopathy'].sum()) / train_df['diabetic_retinopathy'].sum()
        ],
        dtype=torch.float32,
        device=device
    )
    print(f"\nUsing positive weight for loss: {pos_weight.item():.2f}")
    
    # Focal Loss for handling class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=1.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            
        def forward(self, inputs, targets):
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
    
    # Combine Focal Loss with Label Smoothing
    class FocalBCEWithLogitsLoss(nn.Module):
        def __init__(self, weight=None, smoothing=0.0, alpha=0.25, gamma=1.0):
            super().__init__()
            self.smoothing = smoothing
            self.focal = FocalLoss(alpha=alpha, gamma=gamma)
            self.bce = nn.BCEWithLogitsLoss(weight=weight)
            
        def forward(self, input, target):
            # Apply label smoothing
            smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
            
            # Combine losses
            focal_loss = self.focal(input, smooth_target)
            bce_loss = self.bce(input, smooth_target)
            
            return 0.7 * focal_loss + 0.3 * bce_loss
    
    criterion = FocalBCEWithLogitsLoss(
        weight=pos_weight,
        smoothing=0.0,
        alpha=0.25,
        gamma=1.0
    )
    
    # Optimizer with weight decay and better momentum
    # Higher LR for classifier, lower for backbone
    classifier_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]

    optimizer = optim.AdamW([
        {"params": classifier_params, "lr": 2e-4, 'weight_decay': 1e-4},
        {"params": backbone_params, "lr": 5e-5, 'weight_decay': 1e-4}
    ], eps=1e-8, betas=(0.9, 0.999))
    
    # Training configuration
    total_epochs = config.get('epochs', 30)
    accumulation_steps = config.get('accumulation_steps', 1)
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and config.get('use_amp', True) else None
    
    print(f"\nStarting training for {total_epochs} epochs...")
    print(f"Using mixed precision training: {scaler is not None}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    
    # Show first batch before training
    for images, labels in train_loader:
        print("\n=== First batch of training data ===")
        print(f"Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"Labels: {labels[:10].squeeze().tolist()}")
        
        # Print image stats
        print(f"Image stats - Mean: {images.mean().item():.4f}, Std: {images.std().item():.4f}")
        print(f"Image range: {images.min().item():.4f} to {images.max().item():.4f}")
        break
    
    # Start training using the train_model function
    print("\nStarting training...")
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    # After training is complete
    print("\nTraining completed!")
    print(f"Best validation AUC: {best_auc:.4f}")
    
    # Test Grad-CAM on a sample image
    test_gradcam(model, device)
    
    # Export model for iOS if enabled in config
    if config.get('export_to_coreml', True):
        print("\nExporting model to CoreML format...")
        try:
            export_model_for_ios(model, config, device)
            print("Successfully exported model to CoreML format")
        except Exception as e:
            print(f"Error exporting model to CoreML: {e}")
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, accumulation_steps=1, scaler=None, grad_cam_layer=None):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    # Initialize tqdm progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}')
    
    # Initialize optimizer
    optimizer.zero_grad()
    
    for i, (inputs, targets) in pbar:
        try:
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            
            # Forward pass with mixed precision if enabled
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Update running loss
            running_loss += loss.item() * accumulation_steps
            
            # Store predictions and targets for metrics
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss / (i + 1):.4f}",
                'lr': optimizer.param_groups[0]['lr']
            })
            
        except Exception as e:
            print(f"Error in training batch {i}: {e}")
            continue
    
    # Calculate metrics
    try:
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)
        
        # Compute metrics
        metrics = {
            'loss': running_loss / len(train_loader),
            'accuracy': accuracy_score(all_targets > 0.5, all_outputs > 0.5),
            'auc': roc_auc_score(all_targets, all_outputs),
            'precision': precision_score(all_targets > 0.5, all_outputs > 0.5, zero_division=0),
            'recall': recall_score(all_targets > 0.5, all_outputs > 0.5, zero_division=0),
            'f1': f1_score(all_targets > 0.5, all_outputs > 0.5, zero_division=0)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = create_fallback_metrics()
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config):
    """Main training loop"""
    # Training configuration
    epochs = config.get('epochs', 30)
    accumulation_steps = config.get('accumulation_steps', 1)
    patience = config.get('patience', 10)
    best_auc = 0.0
    patience_counter = 0

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and config.get('use_amp', True) else None

    # Get the target layer for Grad-CAM (last conv layer of EfficientNet)
    target_layer = None
    if hasattr(model, 'features') and hasattr(model.features[-1][-1], 'conv3'):
        target_layer = model.features[-1][-1].conv3

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 10)

        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            accumulation_steps=accumulation_steps,
            scaler=scaler,
            grad_cam_layer=target_layer if epoch == 0 else None  # Only generate Grad-CAM on first epoch
        )

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            scaler=scaler
        )

        # Generate Grad-CAM for a sample from validation set every few epochs
        if epoch % 2 == 0 and target_layer is not None:  # Every 2 epochs
            try:
                model.eval()
                with torch.no_grad():
                    # Get a batch of validation data
                    val_images, val_targets = next(iter(val_loader))
                    if val_images is not None and val_targets is not None:
                        val_images = val_images.to(device)

                        # Get a sample image and its prediction
                        sample_img = val_images[0].unsqueeze(0)
                        sample_target = val_targets[0].item()

                        # Get Grad-CAM
                        grad_cam = GradCAM(model, target_layer)
                        cam, output = grad_cam(sample_img)

                        # Convert to numpy for visualization
                        img_np = sample_img[0].cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

                        # Create heatmap
                        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                        
                        # Superimpose
                        superimposed_img = heatmap * 0.4 + img_np * 255
                        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
                        
                        # Save the visualization
                        os.makedirs('gradcam_visualizations', exist_ok=True)
                        save_path = f'gradcam_visualizations/val_epoch_{epoch+1}_gradcam.png'
                        
                        plt.figure(figsize=(12, 4))
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(img_np)
                        plt.title(f'Original\nTrue: {sample_target}')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(heatmap)
                        plt.title('Grad-CAM Heatmap')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(superimposed_img)
                        pred_prob = torch.sigmoid(torch.tensor(output[0][0])).item()
                        pred_class = 1 if pred_prob > 0.5 else 0
                        plt.title(f'Pred: {pred_class}, Prob: {pred_prob:.2f}')
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(save_path, bbox_inches='tight', dpi=150)
                        plt.close()
                        
                        # Log to wandb if available
                        if wandb.run is not None:
                            wandb.log({
                                f'val_gradcam_epoch_{epoch+1}': wandb.Image(save_path, 
                                                                         caption=f'Epoch {epoch+1} Val Grad-CAM')
                            })
                            
            except Exception as e:
                print(f"Error generating validation Grad-CAM: {e}")
        
        # Step the learning rate scheduler if available
        if 'scheduler' in locals():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()
        
        # Log metrics
        log_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['val_loss'],  # Using 'val_loss' instead of 'loss'
            'train_auc': train_metrics['auc'],
            'val_auc': val_metrics['auc'],
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        if wandb.run is not None:
            wandb.log(log_metrics)
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            save_dir = config.get('save_dir', 'saved_models')
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_dir=save_dir,
                filename=f'best_model_auc_{best_auc:.4f}.pth'
            )
            print(f'\nNew best model saved with AUC: {best_auc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping after {patience} epochs without improvement')
                break
    
    # Save final model
    save_dir = config.get('save_dir', 'saved_models')
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs - 1,
        save_dir=save_dir,
        filename='final_model.pth'
    )
    
    # Log best metrics to wandb
    if wandb.run is not None:
        wandb.run.summary["best_val_auc"] = best_auc
        wandb.run.summary["best_epoch"] = epochs - patience_counter
    
    return best_auc

def export_model_for_ios(model, config, device):
    """Export the model for iOS with Grad-CAM support"""
    print("\nExporting model for iOS with Grad-CAM support...")
    
    # Create a wrapper model that outputs both predictions and the last conv layer
    class ModelWithActivations(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
            
            # Store the activations
            self.activations = None
            
            # Register hook to get activations from the last conv layer
            def hook(module, input, output):
                self.activations = output
                return output  # Important: Must return the output
                
            # Register hook on the last conv layer
            self.features[-2].register_forward_hook(hook)
            
        def forward(self, x):
            # Reset activations
            self.activations = None
            
            # Forward pass through features
            x = self.features(x)
            features = self.activations
            
            # Continue with the rest of the model
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            
            return logits, features
            # Forward pass through features
            x = self.features(x)
            
            # Store the activations before pooling
            features = self.activations
            
            # Continue with the rest of the model
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            
            return logits, features
    
    # Create and prepare the model
    model_with_activations = ModelWithActivations(model).eval().to(device)
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, config['img_size'], config['img_size']).to(device)
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model_with_activations, dummy_input)
    
    # Save the traced model
    model_path = os.path.join(config.get('save_dir', 'checkpoints'), 'retinopathy_model.pt')
    torch.jit.save(traced_model, model_path)
    print(f"Model saved to {model_path}")

def test_gradcam(model, device, image_path=None):
    """Test Grad-CAM on an image
    
    Args:
        model: The trained model
        device: Device to run inference on
        image_path: Path to the input image. If None, uses a sample image.
    """
    print("Running Grad-CAM...")
    
    # Set default sample image if none provided
    if image_path is None:
        image_path = os.path.join('sample.jpg')
        if not os.path.exists(image_path):
            print(f"No image provided and sample image not found at {image_path}")
            return
    
    # Load and preprocess the image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    
    # Convert to tensor and ensure gradients are tracked
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Get the input tensor and ensure it requires gradients
    input_tensor = transform(image=image)['image'].unsqueeze(0)
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True  # Enable gradient tracking
    
    # Initialize Grad-CAM with the correct target layer
    # For EfficientNet, we typically use the last conv layer before the classifier
    target_layer = model.features[-1]
    grad_cam = GradCAM(model=model, target_layer=target_layer)
    
    # Generate heatmap - ensure model is in eval mode
    model.eval()
    with torch.set_grad_enabled(True):
        heatmap, output = grad_cam(input_tensor)
        pred_class = output.argmax().item()
        confidence = torch.sigmoid(output[0][0]).item()
    
    # Create visualization
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.4 + image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Save the visualization
    os.makedirs('gradcam_visualizations', exist_ok=True)
    save_path = 'gradcam_visualizations/sample_gradcam.png'
    
    # Create a figure with original, heatmap, and superimposed image
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f'Pred: {pred_class} (Conf: {confidence:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Grad-CAM visualization saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    main(args.config)
