import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pandas as pd
import yaml
import cv2
import psutil
import time
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from gradcam import GradCAM

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Memory monitoring
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB"

# Custom dataset for APTOS fundus images
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test
        
        # Convert multi-class to binary (0: No DR, 1: Any DR)
        if not is_test and 'diagnosis' in self.df.columns:
            self.labels = (self.df['diagnosis'] > 0).astype(int).values
        else:
            self.labels = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = f"{self.df.iloc[idx]['id_code']}.png"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented['image']
                
            if self.is_test or self.labels is None:
                return image
            else:
                label = self.labels[idx]
                return image, torch.tensor([label], dtype=torch.float32)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            blank_image = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transforms:
                blank_image = self.transforms(image=blank_image)['image']
            if self.is_test or self.labels is None:
                return blank_image
            else:
                return blank_image, torch.tensor([0], dtype=torch.float32)

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
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(config['img_size'], config['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def compute_metrics(targets, outputs, threshold=0.5):
    preds = (outputs > threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0)
    }
    try:
        metrics['auc'] = roc_auc_score(targets, outputs)
    except ValueError:
        metrics['auc'] = 0.5
    return metrics

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                   scaler=None, grad_cam_layer=None, grad_accum_steps=1, scheduler=None):
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    # Memory cleanup
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    # Progress bar with memory usage
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    
    for step, (images, targets) in enumerate(pbar):
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if available
        with torch.cuda.amp.autocast(enabled=scaler is not None and device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets) / grad_accum_steps  # Scale loss for gradient accumulation
        
        # Backward pass with gradient scaling if using mixed precision
        if scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step the optimizer every grad_accum_steps
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            if scaler is not None and device.type == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Step the scheduler if it's OneCycleLR
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
            
            optimizer.zero_grad()
            
            # Memory cleanup
            if device.type == 'mps' and (step + 1) % (grad_accum_steps * 10) == 0:
                torch.mps.empty_cache()
        
        # Update running loss (scaled back up for reporting)
        running_loss += loss.item() * images.size(0) * grad_accum_steps
        
        # Store outputs and targets for metrics calculation
        with torch.no_grad():
            all_outputs.extend(outputs.sigmoid().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        if step % 10 == 0 or (step + 1) == len(train_loader):
            pbar.set_postfix({
                'loss': f"{running_loss / ((step + 1) * images.size(0)):.4f}",
                'mem': log_memory_usage().split(': ')[1]
            })
    
    # Calculate metrics
    metrics = compute_metrics(np.array(all_targets), np.array(all_outputs))
    metrics['loss'] = running_loss / len(train_loader.dataset)
    
    # Memory cleanup
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    return metrics

def validate(model, val_loader, criterion, device, scaler=None):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Update running loss
            running_loss += loss.item() * images.size(0)
            
            # Store outputs and targets for metrics calculation
            all_outputs.extend(outputs.sigmoid().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    metrics = compute_metrics(np.array(all_targets), np.array(all_outputs))
    metrics['loss'] = running_loss / len(val_loader.dataset)
    
    return metrics

def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, filename))

def main(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set random seed and device
    set_seed(config.get('seed', 42))
    
    # Set device - prefer MPS (Metal) if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"Initial {log_memory_usage()}")
    
    # Initialize wandb
    wandb.init(project=config.get('wandb_project', 'diabetic_retinopathy'), 
               name=config.get('run_name', 'aptos_binary'),
               config=config)
    
    # Load APTOS data
    train_df = pd.read_csv(os.path.join(config['data_dir'], 'train.csv'))
    
    # Create binary labels (0: No DR, 1: Any DR)
    train_df['diagnosis'] = (train_df['diagnosis'] > 0).astype(int)
    
    # Stratified split
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['diagnosis']
    )
    
    # Print class distribution
    print("\nTraining set class distribution:")
    print(train_df['diagnosis'].value_counts().to_dict())
    print("\nValidation set class distribution:")
    print(val_df['diagnosis'].value_counts().to_dict())
    
    print(f"After loading data: {log_memory_usage()}")
    
    # Create datasets with optimized parameters
    train_ds = APTOSDataset(
        train_df, 
        os.path.join(config['data_dir'], 'train_images'),
        transforms=get_transforms('train', config)
    )
    
    val_ds = APTOSDataset(
        val_df, 
        os.path.join(config['data_dir'], 'train_images'),
        transforms=get_transforms('val', config)
    )
    
    print(f"After creating datasets: {log_memory_usage()}")
    
    # Calculate class weights for imbalanced dataset
    class_counts = train_df['diagnosis'].value_counts().to_dict()
    total = sum(class_counts.values())
    class_weights = {cls: total/(len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[cls] for cls in train_df['diagnosis']]
    
    print(f"After calculating weights: {log_memory_usage()}")
    
    # Optimized dataloaders for MPS
    num_workers = min(4, os.cpu_count() - 1)  # Leave one core free
    batch_size = min(config.get('batch_size', 16), 16)  # Smaller batch size for MPS
    
    print(f"Using {num_workers} workers and batch size {batch_size}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        ),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    print(f"After creating dataloaders: {log_memory_usage()}")
    
    print(f"\nUsing device: {device}")
    print(f"Before model init: {log_memory_usage()}")
    
    # Initialize model
    model = models.efficientnet_b4(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 1024),
        nn.BatchNorm1d(1024),
        nn.SiLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 1, bias=False)
    )
    model = model.to(device)
    
    # Initialize classifier with better initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.classifier.apply(init_weights)
    print(f"After weight init: {log_memory_usage()}")
    
    # Loss function with label smoothing
    class SmoothBCEWithLogitsLoss(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            
        def forward(self, input, target):
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing
            return F.binary_cross_entropy_with_logits(input, target)
    
    criterion = SmoothBCEWithLogitsLoss(smoothing=0.1)
    
    # Optimizer with gradient clipping
    optimizer = optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': config.get('lr', 1e-4) * 0.1},
            {'params': model.classifier.parameters(), 'lr': config.get('lr', 1e-4)},
        ],
        weight_decay=1e-4
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get('lr', 1e-4),
        steps_per_epoch=len(train_loader),
        epochs=config.get('epochs', 30),
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos',
        final_div_factor=1000,
    )
    
    # Mixed precision training (only for CUDA, not for MPS)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Gradient accumulation steps to simulate larger batch size
    grad_accum_steps = max(1, 32 // config.get('batch_size', 16))
    print(f"Using gradient accumulation with {grad_accum_steps} steps")
    
    # Training loop setup
    best_auc = 0.0
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # Memory cleanup
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Before training: {log_memory_usage()}")
    print(f"Starting training for {config.get('epochs', 30)} epochs...")
    
    # Get the target layer for Grad-CAM (last conv layer of EfficientNet)
    target_layer = None
    if hasattr(model, 'features'):
        # For newer torchvision versions, find the last conv layer
        for module in reversed(list(model.features.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
    
    for epoch in range(config.get('epochs', 30)):
        print(f'\nEpoch {epoch + 1}/{config.get("epochs", 30)}')
        print('-' * 10)
        
        # Train for one epoch with gradient accumulation
        optimizer.zero_grad()
        
        # Memory cleanup at start of epoch
        if epoch > 0 and device.type == 'mps':
            torch.mps.empty_cache()
        
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            grad_cam_layer=target_layer if epoch == 0 else None,
            grad_accum_steps=grad_accum_steps,
            scheduler=scheduler if 'OneCycleLR' in str(type(scheduler)) else None
        )
        
        # Step the scheduler (if not OneCycleLR)
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            scaler=scaler
        )
        
        # Generate Grad-CAM for a sample from validation set every 2 epochs
        if epoch % 2 == 0 and target_layer is not None:
            try:
                model.eval()
                with torch.no_grad():
                    # Get a batch of validation data
                    val_images, val_targets = next(iter(val_loader))
                    val_images = val_images.to(device)
                    
                    # Get a sample image and its prediction
                    sample_img = val_images[0].unsqueeze(0)
                    sample_target = val_targets[0].item()
                    
                    # Get Grad-CAM
                    grad_cam = GradCAM(model, target_layer)
                    cam, output = grad_cam(sample_img)
                    pred_class = (output.sigmoid() > 0.5).int().item()
                    
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
                    plt.title(f'Pred: {pred_class}, Prob: {output.sigmoid()[0][0]:.2f}')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                    
                    # Log to wandb if available
                    if wandb.run is not None:
                        wandb.log({
                            f'gradcam_epoch_{epoch+1}': wandb.Image(save_path, caption=f'Epoch {epoch+1} Grad-CAM')
                        })
            except Exception as e:
                print(f"Error generating validation Grad-CAM: {e}")
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        metrics = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'train_auc': train_metrics['auc'],
            'val_auc': val_metrics['auc'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        print(f"Train Loss: {metrics['train_loss']:.4f}, Val Loss: {metrics['val_loss']:.4f}")
        print(f"Train Acc: {metrics['train_accuracy']:.4f}, Val Acc: {metrics['val_accuracy']:.4f}")
        print(f"Train AUC: {metrics['train_auc']:.4f}, Val AUC: {metrics['val_auc']:.4f}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log(metrics)
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_model_path = os.path.join(save_dir, f'best_model_auc_{best_auc:.4f}.pth')
            save_checkpoint(model, optimizer, epoch, save_dir, f'best_model_auc_{best_auc:.4f}.pth')
            print(f'\nNew best model saved with AUC: {best_auc:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, 
                save_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    print("\nTraining completed!")
    print(f"Best validation AUC: {best_auc:.4f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.get('epochs', 30) - 1,
        save_dir, 'final_model.pth'
    )
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Create a default config if it doesn't exist
    config_path = args.config
    if not os.path.exists(config_path):
        config = {
            'data_dir': '/Users/Ronit/Downloads/aptos2019-blindness-detection',
            'img_size': 224,
            'batch_size': 32,
            'epochs': 30,
            'lr': 1e-4,
            'wandb_project': 'diabetic_retinopathy',
            'run_name': 'aptos_binary_classification',
            'save_dir': 'checkpoints'
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Created default config at {config_path}")
    
    main(config_path)
