import torch
import yaml
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models
from utils.dataset import FundusDataset, get_transforms
from utils.metrics import compute_metrics, print_metrics
from utils.utils import load_checkpoint


def main(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load test data
    test_df = pd.read_csv(config['test_csv'])
    
    # Print first few rows to verify format
    print("Test data sample:")
    print(test_df.head())
    
    # Create dataset and dataloader
    test_ds = FundusDataset(test_df, config['img_dir'], get_transforms('val', config))
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    print(f"Loaded {len(test_ds)} test images")

    # Initialize model (using the same architecture as training)
    model = models.efficientnet_b4(weights=None)  # We'll load our own weights
    
    # Define the custom classifier head (must match training architecture)
    num_ftrs = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(num_ftrs, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.SiLU(inplace=True),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.SiLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 1, bias=False)
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    
    # Load the checkpoint
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Print available keys in checkpoint for debugging
    print("Available keys in checkpoint:", checkpoint.keys())
    
    # Initialize model with the same architecture as training
    model = models.efficientnet_b4(weights=None)
    
    # Define the exact same classifier head as in training
    num_ftrs = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(num_ftrs, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.SiLU(inplace=True),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.SiLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 1, bias=False)
    )
    
    # Move model to device
    model = model.to(device)
    
    # Load the model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # If no specific state dict key, assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Run evaluation
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device).float()
                
                # Get model predictions
                outputs = model(images)
                
                # Handle single output case
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                    
                # Store predictions and targets
                all_outputs.append(outputs.sigmoid().cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    if len(all_outputs) == 0:
        raise ValueError("No predictions were made. Check your data loading.")
        
    all_outputs = np.concatenate(all_outputs).reshape(-1)
    all_targets = np.concatenate(all_targets).reshape(-1)
    metrics = compute_metrics(all_targets, all_outputs)
    print_metrics(metrics, ['No DR', 'DR'])  # Assuming binary classification

    # Confusion matrix
    preds = (all_outputs > 0.5).astype(int)
    cm = confusion_matrix(all_targets, preds)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No DR', 'DR'], 
                yticklabels=['No DR', 'DR'],
                annot_kws={"size": 14})
    plt.title('Confusion Matrix - Diabetic Retinopathy Classification', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    output_path = 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved as: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    main(args.config, args.checkpoint)
