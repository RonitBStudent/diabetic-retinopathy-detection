import torch
import yaml
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, 
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score
)
from torch.utils.data import DataLoader
from torchvision import models
from utils.dataset import FundusDataset, get_transforms
from utils.metrics import compute_metrics, print_metrics
import cv2

def load_model_and_data(config_path, checkpoint_path):
    """Load model and test data"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load test data
    test_df = pd.read_csv(config['test_csv'])
    test_ds = FundusDataset(test_df, config['img_dir'], get_transforms('val', config))
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    print(f"Loaded {len(test_ds)} test images")

    # Initialize model
    model = models.efficientnet_b4(weights=None)
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, test_loader, device, test_df

def get_predictions(model, test_loader, device):
    """Get model predictions and true labels"""
    all_outputs = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device).float()
                
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                    
                all_outputs.append(outputs.sigmoid().cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_images.append(images.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs).reshape(-1)
    all_targets = np.concatenate(all_targets).reshape(-1)
    all_images = np.concatenate(all_images)
    
    return all_outputs, all_targets, all_images

def create_confusion_matrix_plot(y_true, y_pred, save_path):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No DR', 'DR'], 
                yticklabels=['No DR', 'DR'],
                annot_kws={"size": 16}, square=True)
    plt.title('Confusion Matrix\nDiabetic Retinopathy Classification', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # Add metrics text
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    plt.text(1.05, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curve_plot(y_true, y_scores, save_path):
    """Create ROC curve visualization"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve\nDiabetic Retinopathy Classification', fontsize=18, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_precision_recall_curve_plot(y_true, y_scores, save_path):
    """Create Precision-Recall curve visualization"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve\nDiabetic Retinopathy Classification', fontsize=18, pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add baseline for random classifier
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label=f'Random Classifier ({baseline:.3f})')
    plt.legend(loc="lower left", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_distribution_plot(y_scores, y_true, save_path):
    """Create prediction score distribution plot"""
    plt.figure(figsize=(12, 8))
    
    # Separate scores for each class
    no_dr_scores = y_scores[y_true == 0]
    dr_scores = y_scores[y_true == 1]
    
    plt.hist(no_dr_scores, bins=50, alpha=0.7, label='No DR', color='skyblue', density=True)
    plt.hist(dr_scores, bins=50, alpha=0.7, label='DR', color='salmon', density=True)
    
    plt.xlabel('Prediction Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Prediction Score Distribution\nDiabetic Retinopathy Classification', fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add threshold line
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold = 0.5')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_summary_plot(y_true, y_scores, save_path):
    """Create comprehensive metrics summary plot"""
    # Calculate metrics at different thresholds
    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=8)
    plt.plot(thresholds, precisions, 's-', label='Precision', linewidth=2, markersize=8)
    plt.plot(thresholds, recalls, '^-', label='Recall', linewidth=2, markersize=8)
    plt.plot(thresholds, f1_scores, 'd-', label='F1-Score', linewidth=2, markersize=8)
    
    plt.xlabel('Classification Threshold', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.title('Performance Metrics vs Classification Threshold\nDiabetic Retinopathy Classification', fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.1, 0.9])
    plt.ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_sample_predictions_plot(images, y_true, y_scores, y_pred, save_path, num_samples=8):
    """Create sample predictions visualization"""
    plt.figure(figsize=(16, 12))
    
    # Select samples (mix of correct and incorrect predictions)
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]
    
    # Mix of correct and incorrect predictions
    selected_indices = []
    if len(correct_indices) > 0:
        selected_indices.extend(np.random.choice(correct_indices, min(4, len(correct_indices)), replace=False))
    if len(incorrect_indices) > 0:
        selected_indices.extend(np.random.choice(incorrect_indices, min(4, len(incorrect_indices)), replace=False))
    
    # If we don't have enough variety, just take random samples
    if len(selected_indices) < num_samples:
        remaining = num_samples - len(selected_indices)
        available = list(set(range(len(images))) - set(selected_indices))
        if available:
            selected_indices.extend(np.random.choice(available, min(remaining, len(available)), replace=False))
    
    for i, idx in enumerate(selected_indices[:num_samples]):
        plt.subplot(2, 4, i+1)
        
        # Denormalize image (assuming ImageNet normalization)
        img = images[idx].transpose(1, 2, 0)
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        
        true_label = "No DR" if y_true[idx] == 0 else "DR"
        pred_label = "No DR" if y_pred[idx] == 0 else "DR"
        confidence = y_scores[idx]
        correct = y_true[idx] == y_pred[idx]
        
        color = 'green' if correct else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}', 
                 fontsize=10, color=color)
    
    plt.suptitle('Sample Predictions\n(Green=Correct, Red=Incorrect)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    config_path = "configs/aptos_binary_m4.yaml"
    checkpoint_path = "/Users/Ronit/Desktop/CascadeProjects/fundus_disease_detection/Binary Classification Model/best_model_auc_0.9971.pth copy"
    
    # Load model and data
    model, test_loader, device, test_df = load_model_and_data(config_path, checkpoint_path)
    
    # Get predictions
    y_scores, y_true, all_images = get_predictions(model, test_loader, device)
    y_pred = (y_scores > 0.5).astype(int)
    
    # Create output directory
    os.makedirs('research_plots', exist_ok=True)
    
    # Generate all plots
    print("Generating research-quality visualizations...")
    
    create_confusion_matrix_plot(y_true, y_pred, 'research_plots/confusion_matrix.png')
    print("✓ Confusion matrix saved")
    
    create_roc_curve_plot(y_true, y_scores, 'research_plots/roc_curve.png')
    print("✓ ROC curve saved")
    
    create_precision_recall_curve_plot(y_true, y_scores, 'research_plots/precision_recall_curve.png')
    print("✓ Precision-Recall curve saved")
    
    create_prediction_distribution_plot(y_scores, y_true, 'research_plots/prediction_distribution.png')
    print("✓ Prediction distribution saved")
    
    create_metrics_summary_plot(y_true, y_scores, 'research_plots/metrics_summary.png')
    print("✓ Metrics summary saved")
    
    create_sample_predictions_plot(all_images, y_true, y_scores, y_pred, 'research_plots/sample_predictions.png')
    print("✓ Sample predictions saved")
    
    # Print final metrics
    metrics = compute_metrics(y_true, y_scores)
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("="*50)
    print(f"\nAll visualizations saved in 'research_plots/' directory!")

if __name__ == "__main__":
    main()
