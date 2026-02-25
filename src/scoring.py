import torch
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def diffusion_classifier_score(
    model,
    scheduler,
    images: torch.Tensor,
    num_conditions: int = 2,
    num_trials: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Algorithm 1: Diffusion Classifier for OOD scoring.
    
    For each image, sample random timesteps and noise, then measure
    reconstruction error under each class conditioning. The class with
    lower error is predicted as the image's class. OOD score is the
    error under the ID condition (c=0).
    
    Args:
        model: Conditional UNet model
        scheduler: DDPM scheduler with add_noise method
        images: Input images [B, C, H, W]
        num_conditions: Number of class conditions
        num_trials: Number of random trials per image
        
    Returns:
        ood_scores: Reconstruction error under c=0 [B]
        predictions: Predicted class per image [B]
    """
    device = images.device
    batch_size = images.shape[0]
    num_timesteps = scheduler.config.num_train_timesteps
    
    all_errors = torch.zeros(batch_size, num_conditions, num_trials, device=device)
    
    for trial_idx in range(num_trials):
        timesteps = torch.randint(1, num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(images)
        
        noisy = scheduler.add_noise(images, noise, timesteps)
        
        for c in range(num_conditions):
            labels = torch.full((batch_size,), c, device=device, dtype=torch.long)
            
            pred = model(noisy, timesteps, class_labels=labels)
            if hasattr(pred, 'sample'):
                pred = pred.sample
            
            mse = F.mse_loss(pred, noise, reduction='none').mean(dim=[1, 2, 3])
            all_errors[:, c, trial_idx] = mse
    
    mean_errors = all_errors.mean(dim=2)
    predictions = mean_errors.argmin(dim=1)
    ood_scores = mean_errors[:, 0]
    
    return ood_scores, predictions


@torch.no_grad()
def compute_per_timestep_errors(
    model,
    scheduler,
    images: torch.Tensor,
    timesteps_to_eval: list = None,
    condition: int = 0,
) -> dict:
    """
    Analyze reconstruction error across different timesteps.
    
    Useful for understanding how the model behaves at different noise levels.
    """
    device = images.device
    batch_size = images.shape[0]
    
    if timesteps_to_eval is None:
        timesteps_to_eval = [50, 100, 200, 300, 500, 700, 900]
    
    results = {}
    labels = torch.full((batch_size,), condition, device=device, dtype=torch.long)
    
    for t in timesteps_to_eval:
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise = torch.randn_like(images)
        
        noisy = scheduler.add_noise(images, noise, t_batch)
        
        pred = model(noisy, t_batch, class_labels=labels)
        if hasattr(pred, 'sample'):
            pred = pred.sample
        
        mse = F.mse_loss(pred, noise, reduction='none').mean(dim=[1, 2, 3])
        results[t] = {
            'mean': mse.mean().item(),
            'std': mse.std().item(),
            'values': mse.cpu(),
        }
    
    return results


@torch.no_grad()
def score_dataset(
    model,
    scheduler,
    dataloader,
    num_conditions: int = 2,
    num_trials: int = 10,
    device: torch.device = None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Score an entire dataset using the diffusion classifier.
    
    Returns:
        all_scores: OOD scores for all samples
        all_predictions: Predicted classes
        all_labels: Ground truth binary labels
    """
    if device is None:
        device = next(model.parameters()).device
    
    all_scores = []
    all_predictions = []
    all_labels = []
    
    iterator = tqdm(dataloader, desc="Scoring") if show_progress else dataloader
    
    for batch in iterator:
        images = batch["images"].to(device)
        labels = batch["binary_labels"]
        
        scores, preds = diffusion_classifier_score(
            model, scheduler, images,
            num_conditions=num_conditions,
            num_trials=num_trials,
        )
        
        all_scores.append(scores.cpu())
        all_predictions.append(preds.cpu())
        all_labels.append(labels)
    
    return (
        torch.cat(all_scores),
        torch.cat(all_predictions),
        torch.cat(all_labels),
    )
