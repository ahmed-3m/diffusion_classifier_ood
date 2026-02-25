import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present, AUROC undefined")
        return 0.5
    return roc_auc_score(labels, scores)


def compute_fpr_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_tpr: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute False Positive Rate at a target True Positive Rate.
    
    Returns:
        fpr: FPR value at target TPR
        threshold: Score threshold achieving this TPR
    """
    if len(np.unique(labels)) < 2:
        return 1.0, 0.0
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0, thresholds[-1]
    
    return fpr[idx[0]], thresholds[idx[0]]


def compute_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return average_precision_score(labels, scores)


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy."""
    return (predictions == labels).mean()


def compute_det_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Detection Error Tradeoff curve.
    
    Returns False Negative Rate vs False Positive Rate.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    return fpr, fnr


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute all OOD detection metrics.
    
    Returns dict with: auroc, fpr95, aupr, accuracy (if predictions given)
    """
    metrics = {}
    
    metrics['auroc'] = compute_auroc(labels, scores)
    metrics['fpr95'], metrics['threshold_95'] = compute_fpr_at_tpr(labels, scores, 0.95)
    metrics['aupr'] = compute_aupr(labels, scores)
    
    if predictions is not None:
        metrics['accuracy'] = compute_accuracy(predictions, labels)
        
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable table string."""
    lines = ["=" * 40]
    lines.append(" OOD Detection Metrics")
    lines.append("=" * 40)
    
    key_display = {
        'auroc': 'AUROC',
        'fpr95': 'FPR@95%TPR',
        'aupr': 'AUPR',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
    }
    
    for key, display in key_display.items():
        if key in metrics:
            lines.append(f" {display:15s}: {metrics[key]:.4f}")
    
    lines.append("=" * 40)
    return "\n".join(lines)
