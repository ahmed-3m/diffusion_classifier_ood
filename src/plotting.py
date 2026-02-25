import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from typing import Optional, Tuple, List
import torch
import wandb
import logging

logger = logging.getLogger(__name__)


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    auroc: float = None,
) -> plt.Figure:
    """Plot ROC curve with AUROC annotation."""
    fpr, tpr, _ = roc_curve(labels, scores)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.2)
    
    if auroc is not None:
        ax.text(0.6, 0.2, f'AUROC = {auroc:.4f}', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_precision_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    aupr: float = None,
) -> plt.Figure:
    """Plot Precision-Recall curve with AUPR annotation."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.fill_between(recall, precision, alpha=0.2)
    
    if aupr is not None:
        ax.text(0.2, 0.2, f'AUPR = {aupr:.4f}', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_score_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> plt.Figure:
    """Plot overlapping histograms of ID and OOD score distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = np.linspace(
        min(id_scores.min(), ood_scores.min()),
        max(id_scores.max(), ood_scores.max()),
        50
    )
    
    ax.hist(id_scores, bins=bins, alpha=0.6, label='ID (c=0)', color='#2ecc71', density=True)
    ax.hist(ood_scores, bins=bins, alpha=0.6, label='OOD (c=1)', color='#e74c3c', density=True)
    
    ax.axvline(id_scores.mean(), color='#27ae60', linestyle='--', label=f'ID mean: {id_scores.mean():.3f}')
    ax.axvline(ood_scores.mean(), color='#c0392b', linestyle='--', label=f'OOD mean: {ood_scores.mean():.3f}')
    
    ax.set_xlabel('OOD Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_score_violin(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> plt.Figure:
    """Violin plot of score distributions."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    parts = ax.violinplot([id_scores, ood_scores], positions=[1, 2], showmeans=True, showmedians=True)
    
    parts['bodies'][0].set_facecolor('#2ecc71')
    parts['bodies'][1].set_facecolor('#e74c3c')
    for pc in parts['bodies']:
        pc.set_alpha(0.6)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ID (c=0)', 'OOD (c=1)'])
    ax.set_ylabel('OOD Score')
    ax.set_title('Score Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_det_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> plt.Figure:
    """Plot Detection Error Tradeoff curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, fnr, 'b-', linewidth=2)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_title('Detection Error Tradeoff')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_fpr_vs_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_95: float = None,
) -> plt.Figure:
    """Plot FPR and TPR vs threshold with TPR95 point marked."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # roc_curve returns thresholds with len = len(fpr) - 1
    # Align arrays by using the first len(thresholds) values
    n = len(thresholds)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, fpr[:n], 'r-', label='FPR', linewidth=2)
    ax.plot(thresholds, tpr[:n], 'g-', label='TPR', linewidth=2)
    
    if threshold_95 is not None:
        ax.axvline(threshold_95, color='gray', linestyle='--', alpha=0.7, label=f'TPR95 threshold: {threshold_95:.3f}')
    
    ax.axhline(0.95, color='green', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('FPR and TPR vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_timestep_error(
    timestep_results: dict,
) -> plt.Figure:
    """Plot reconstruction error vs timestep."""
    timesteps = sorted(timestep_results.keys())
    means = [timestep_results[t]['mean'] for t in timesteps]
    stds = [timestep_results[t]['std'] for t in timesteps]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(timesteps, means, yerr=stds, fmt='o-', capsize=4, capthick=2)
    
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Mean Reconstruction Error')
    ax.set_title('Reconstruction Error vs Timestep')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str] = None,
) -> plt.Figure:
    """Plot confusion matrix for binary classification."""
    if class_names is None:
        class_names = ['ID (c=0)', 'OOD (c=1)']
    
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                          color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    fig.colorbar(im)
    plt.tight_layout()
    return fig


def plot_tsne_embeddings(
    scores: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
) -> plt.Figure:
    """t-SNE visualization of reconstruction errors."""
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    
    if scores.shape[1] == 1:
        noise = np.random.randn(len(scores), 5) * 0.1
        scores_aug = np.hstack([scores, noise])
    else:
        scores_aug = scores
    
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(scores) - 1), random_state=42)
    embedded = tsne.fit_transform(scores_aug)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    id_mask = labels == 0
    ood_mask = labels == 1
    
    ax.scatter(embedded[id_mask, 0], embedded[id_mask, 1], c='#2ecc71', label='ID', alpha=0.6, s=20)
    ax.scatter(embedded[ood_mask, 0], embedded[ood_mask, 1], c='#e74c3c', label='OOD', alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE of Reconstruction Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_generated_samples(
    samples: torch.Tensor,
    title: str = "Generated Samples",
    ncols: int = 8,
) -> plt.Figure:
    """Plot a grid of generated samples."""
    samples = samples.cpu()
    if samples.min() < 0:
        samples = samples * 0.5 + 0.5
    samples = samples.clamp(0, 1)
    
    n = len(samples)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')
        if i < n:
            img = samples[i].permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_comparison_grid(
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    title: str = "Real vs Generated",
) -> plt.Figure:
    """Side-by-side comparison of real and generated images."""
    n = min(len(real_samples), len(generated_samples), 8)
    
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    
    for i in range(n):
        real = real_samples[i].cpu()
        gen = generated_samples[i].cpu()
        
        if real.min() < 0:
            real = real * 0.5 + 0.5
        if gen.min() < 0:
            gen = gen * 0.5 + 0.5
        
        axes[0, i].imshow(real.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, i].axis('off')
        
        axes[1, i].imshow(gen.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, i].axis('off')
        
        if i == 0:
            axes[0, i].set_title('Real', fontsize=10)
            axes[1, i].set_title('Generated', fontsize=10)
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_extreme_samples(
    images: torch.Tensor,
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 4,
) -> plt.Figure:
    """Show images with highest/lowest confidence."""
    id_mask = labels == 0
    ood_mask = labels == 1
    
    id_scores = scores[id_mask]
    id_images = images[id_mask]
    ood_scores = scores[ood_mask]
    ood_images = images[ood_mask]
    
    id_sorted = np.argsort(id_scores)
    ood_sorted = np.argsort(ood_scores)[::-1]
    
    fig, axes = plt.subplots(2, k, figsize=(k * 2, 4.5))
    
    for i in range(k):
        if i < len(id_sorted):
            idx = id_sorted[i]
            img = id_images[idx].cpu()
            if img.min() < 0:
                img = img * 0.5 + 0.5
            axes[0, i].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[0, i].set_title(f'{id_scores[idx]:.3f}', fontsize=9)
        axes[0, i].axis('off')
        
        if i < len(ood_sorted):
            idx = ood_sorted[i]
            img = ood_images[idx].cpu()
            if img.min() < 0:
                img = img * 0.5 + 0.5
            axes[1, i].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[1, i].set_title(f'{ood_scores[idx]:.3f}', fontsize=9)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Most ID-like', fontsize=10)
    axes[1, 0].set_ylabel('Most OOD-like', fontsize=10)
    
    fig.suptitle('Extreme Score Samples', fontsize=12)
    plt.tight_layout()
    return fig


def log_all_plots_to_wandb(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    metrics: dict,
    prefix: str = "eval",
) -> None:
    """Log all evaluation plots to W&B."""
    try:
        id_mask = labels == 0
        ood_mask = labels == 1
        
        id_scores = scores[id_mask]
        ood_scores = scores[ood_mask]
        
        figs = {}
        
        figs[f'{prefix}/roc_curve'] = plot_roc_curve(labels, scores, metrics.get('auroc'))
        figs[f'{prefix}/pr_curve'] = plot_precision_recall(labels, scores, metrics.get('aupr'))
        figs[f'{prefix}/score_histogram'] = plot_score_histogram(id_scores, ood_scores)
        figs[f'{prefix}/score_violin'] = plot_score_violin(id_scores, ood_scores)
        figs[f'{prefix}/det_curve'] = plot_det_curve(labels, scores)
        figs[f'{prefix}/fpr_threshold'] = plot_fpr_vs_threshold(labels, scores, metrics.get('threshold_95'))
        figs[f'{prefix}/confusion_matrix'] = plot_confusion_matrix(labels, predictions)
        
        for name, fig in figs.items():
            wandb.log({name: wandb.Image(fig)})
            plt.close(fig)
            
    except Exception as e:
        logger.warning(f"Failed to log plots to W&B: {e}")
