import torch
import torch.nn.functional as F
from typing import Tuple, Literal
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def sample_weighted_timesteps(
    batch_size: int,
    num_timesteps: int,
    device: torch.device,
    mode: Literal["uniform", "mid_focus", "stratified"] = "mid_focus",
) -> torch.Tensor:
    """
    Sample timesteps with various strategies.
    
    Args:
        batch_size: Number of timesteps to sample
        num_timesteps: Total number of diffusion timesteps
        device: Target device
        mode: Sampling strategy
            - "uniform": Uniform random sampling across all timesteps
            - "mid_focus": Weighted sampling focusing on t=100-500 (best for OOD)
            - "stratified": Stratified sampling across bins
    
    Returns:
        Tensor of timesteps [batch_size]
    """
    if mode == "uniform":
        return torch.randint(1, num_timesteps, (batch_size,), device=device)
    
    elif mode == "mid_focus":
        # Focus on mid-range timesteps where class separation is strongest
        # Use truncated normal distribution centered at t=300
        mean = 300.0
        std = 150.0
        t_min, t_max = 50, 700
        
        # Sample from truncated normal
        samples = torch.randn(batch_size, device=device) * std + mean
        samples = samples.clamp(t_min, t_max).long()
        return samples
    
    elif mode == "stratified":
        # Stratified sampling: divide into bins and sample from each
        bins = [50, 150, 300, 500, 700, 900]
        bin_idx = torch.randint(0, len(bins) - 1, (batch_size,), device=device)
        
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = bin_idx == i
            count = mask.sum().item()
            if count > 0:
                timesteps[mask] = torch.randint(low, high, (count,), device=device)
        return timesteps
    
    else:
        raise ValueError(f"Unknown timestep sampling mode: {mode}")


@torch.no_grad()
def diffusion_classifier_score(
    model,
    scheduler,
    images: torch.Tensor,
    num_conditions: int = 2,
    num_trials: int = 10,
    scoring_method: Literal["difference", "ratio", "id_error"] = "difference",
    timestep_mode: Literal["uniform", "mid_focus", "stratified"] = "mid_focus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Algorithm 1: Diffusion Classifier for OOD scoring.
    
    For each image, sample timesteps and noise, then measure reconstruction
    error under each class conditioning. OOD score is computed based on the
    chosen scoring method.
    
    Args:
        model: Conditional UNet model
        scheduler: DDPM scheduler with add_noise method
        images: Input images [B, C, H, W]
        num_conditions: Number of class conditions
        num_trials: Number of random trials per image
        scoring_method: How to compute OOD score
            - "difference": error(c=0) - error(c=1) [RECOMMENDED]
            - "ratio": error(c=0) / error(c=1)
            - "id_error": error(c=0) only (original method)
        timestep_mode: How to sample timesteps
            - "mid_focus": Focus on t=100-500 [RECOMMENDED]
            - "uniform": Uniform random
            - "stratified": Stratified across ranges
        
    Returns:
        ood_scores: OOD scores for each image [B]
            Higher = more likely OOD
        predictions: Predicted class per image [B]
            0 = predicted ID, 1 = predicted OOD
    """
    device = images.device
    batch_size = images.shape[0]
    num_timesteps = scheduler.config.num_train_timesteps
    
    all_errors = torch.zeros(batch_size, num_conditions, num_trials, device=device)
    
    for trial_idx in range(num_trials):
        # Use weighted timestep sampling
        timesteps = sample_weighted_timesteps(
            batch_size, num_timesteps, device, mode=timestep_mode
        )
        noise = torch.randn_like(images)
        
        noisy = scheduler.add_noise(images, noise, timesteps)
        
        for c in range(num_conditions):
            labels = torch.full((batch_size,), c, device=device, dtype=torch.long)
            
            pred = model(noisy, timesteps, class_labels=labels)
            if hasattr(pred, 'sample'):
                pred = pred.sample
            
            mse = F.mse_loss(pred, noise, reduction='none').mean(dim=[1, 2, 3])
            all_errors[:, c, trial_idx] = mse
    
    # Average across trials
    mean_errors = all_errors.mean(dim=2)
    
    # Predictions based on which condition has lower error
    predictions = mean_errors.argmin(dim=1)
    
    # Compute OOD scores based on chosen method
    if scoring_method == "difference":
        # RECOMMENDED: Difference-based scoring
        # Higher score = higher ID error relative to OOD error = more likely OOD
        ood_scores = mean_errors[:, 0] - mean_errors[:, 1]
    elif scoring_method == "ratio":
        # Ratio-based scoring
        ood_scores = mean_errors[:, 0] / (mean_errors[:, 1] + 1e-8)
    elif scoring_method == "id_error":
        # Original method: just use ID error
        ood_scores = mean_errors[:, 0]
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")
    
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
