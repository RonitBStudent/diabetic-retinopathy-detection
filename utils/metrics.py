import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    return {
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def print_metrics(metrics, class_names):
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
