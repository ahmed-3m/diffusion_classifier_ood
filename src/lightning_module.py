import torch
import torch.nn.functional as F
import diffusers
import lightning as L
from typing import Dict, List, Optional
import logging

from .model import ConditionalUNet
from .scoring import diffusion_classifier_score
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class DiffusionClassifierOOD(L.LightningModule):
    """
    PyTorch Lightning module for training a conditional diffusion model
    as an OOD detector.
    
    The model learns to denoise images conditioned on a binary class label:
        c=0: In-distribution class
        c=1: Out-of-distribution proxy (all other classes)
    
    At inference, OOD detection is performed by comparing reconstruction
    errors across conditions.
    """
    
    def __init__(
        self,
        sample_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        layers_per_block: int = 2,
        block_out_channels: tuple = (128, 256, 256, 256),
        down_block_types: tuple = (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: tuple = (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds: int = 2,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        num_trials: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.validation_outputs: List[Dict] = []
        
        self.model = ConditionalUNet(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=num_class_embeds,
        )
        
        self.scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
        )
        
        logger.info(f"Model initialized with {self.model.get_num_params() / 1e6:.2f}M parameters")
    
    def forward(self, images, timesteps, class_labels):
        return self.model(images, timesteps, class_labels)
    
    def training_step(self, batch, batch_idx):
        images = batch["images"]
        labels = batch["binary_labels"]
        
        timesteps = torch.randint(
            0, self.hparams.num_train_timesteps,
            (images.size(0),), device=self.device
        )
        noise = torch.randn_like(images)
        
        noisy = self.scheduler.add_noise(images, noise, timesteps)
        pred = self.model(noisy, timesteps, labels)
        
        loss = F.mse_loss(pred, noise)
        
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        binary_labels = batch["binary_labels"]
        
        scores, predictions = diffusion_classifier_score(
            self.model,
            self.scheduler,
            images,
            num_conditions=self.hparams.num_class_embeds,
            num_trials=self.hparams.num_trials,
        )
        
        self.validation_outputs.append({
            "scores": scores.cpu(),
            "predictions": predictions.cpu(),
            "labels": binary_labels.cpu(),
        })
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        all_scores = torch.cat([x["scores"] for x in self.validation_outputs]).numpy()
        all_preds = torch.cat([x["predictions"] for x in self.validation_outputs]).numpy()
        all_labels = torch.cat([x["labels"] for x in self.validation_outputs]).numpy()
        
        self.validation_outputs.clear()
        
        metrics = compute_all_metrics(all_labels, all_scores, all_preds)
        
        self.log("val/auroc", metrics['auroc'], prog_bar=True, sync_dist=True)
        self.log("val/fpr95", metrics['fpr95'], prog_bar=False, sync_dist=True)
        self.log("val/aupr", metrics['aupr'], prog_bar=False, sync_dist=True)
        
        if 'accuracy' in metrics:
            self.log("val/accuracy", metrics['accuracy'], prog_bar=True, sync_dist=True)
        
        logger.info(
            f"Epoch {self.current_epoch} | "
            f"AUROC: {metrics['auroc']:.4f} | "
            f"FPR95: {metrics['fpr95']:.4f} | "
            f"Accuracy: {metrics.get('accuracy', 0):.4f}"
        )
    
    @torch.no_grad()
    def sample_images(
        self,
        num_samples: int = 8,
        condition: int = 0,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Generate samples from the model under a specific condition."""
        device = self.device
        sample = torch.randn(num_samples, 3, 32, 32, device=device)
        labels = torch.full((num_samples,), condition, device=device, dtype=torch.long)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = self.model(sample, t_batch, labels)
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample
        
        return sample
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
