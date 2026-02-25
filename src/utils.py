import os
import logging
from glob import glob
from datetime import datetime
from typing import Optional
import torch
import lightning as L
from huggingface_hub import HfApi, create_repo
import matplotlib.pyplot as plt
import wandb


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging format for the project."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def generate_experiment_name(tag: str = "run") -> str:
    """Generate a timestamped experiment name."""
    now = datetime.now()
    return f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{tag}"


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint file."""
    patterns = [
        os.path.join(checkpoint_dir, "last*.ckpt"),
        os.path.join(checkpoint_dir, "*.ckpt"),
    ]
    
    for pattern in patterns:
        candidates = glob(pattern)
        if candidates:
            return max(candidates, key=os.path.getmtime)
    
    return None


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 1,
    keep_best: bool = True,
) -> int:
    """Remove old checkpoint files, keeping only recent ones."""
    all_ckpts = glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not all_ckpts:
        return 0
    
    keep = set()
    
    last_ckpts = [c for c in all_ckpts if 'last' in os.path.basename(c)]
    if last_ckpts:
        last_ckpts.sort(key=os.path.getmtime, reverse=True)
        keep.update(last_ckpts[:keep_last])
    
    if keep_best:
        best_ckpts = [c for c in all_ckpts if 'best' in os.path.basename(c)]
        keep.update(best_ckpts)
    
    to_delete = set(all_ckpts) - keep
    for path in to_delete:
        try:
            os.remove(path)
        except OSError:
            pass
    
    return len(to_delete)


def push_to_huggingface(
    checkpoint_path: str,
    repo_name: str,
    model_card: str = "",
    token: str = None,
) -> str:
    """Upload model checkpoint to HuggingFace Hub."""
    logger = logging.getLogger(__name__)
    
    api = HfApi(token=token)
    
    try:
        create_repo(repo_name, exist_ok=True, token=token)
    except Exception as e:
        logger.warning(f"Could not create repo: {e}")
    
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="model.ckpt",
        repo_id=repo_name,
        token=token,
    )
    
    if model_card:
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token,
        )
    
    url = f"https://huggingface.co/{repo_name}"
    logger.info(f"Model uploaded to {url}")
    
    return url


class MemoryCleanupCallback(L.Callback):
    """Clear GPU memory cache after each epoch."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SampleVisualizationCallback(L.Callback):
    """Periodically generate and log samples from both conditions."""
    
    def __init__(self, every_n_epochs: int = 10, num_samples: int = 8):
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.logger = logging.getLogger(__name__)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        
        pl_module.eval()
        
        try:
            samples_c0 = pl_module.sample_images(self.num_samples, condition=0)
            samples_c1 = pl_module.sample_images(self.num_samples, condition=1)
            
            fig = self._create_grid(samples_c0, samples_c1, trainer.current_epoch)
            
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                try:
                    trainer.logger.experiment.log({
                        "samples/generated": wandb.Image(fig),
                        "epoch": trainer.current_epoch,
                    })
                except Exception:
                    pass
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate samples: {e}")
    
    def _create_grid(self, samples_c0, samples_c1, epoch):
        n = len(samples_c0)
        fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
        
        for i in range(n):
            img0 = samples_c0[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            img1 = samples_c1[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            
            axes[0, i].imshow(img0.clamp(0, 1).numpy())
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img1.clamp(0, 1).numpy())
            axes[1, i].axis('off')
            
            if i == 0:
                axes[0, i].set_title('c=0 (ID)', fontsize=9)
                axes[1, i].set_title('c=1 (OOD)', fontsize=9)
        
        fig.suptitle(f'Epoch {epoch}', fontsize=11)
        plt.tight_layout()
        return fig


class OODEvaluationCallback(L.Callback):
    """Log detailed OOD plots at specified intervals."""
    
    def __init__(self, every_n_epochs: int = 10):
        self.every_n_epochs = every_n_epochs
        self.logger = logging.getLogger(__name__)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        
        pass


class HuggingFaceUploadCallback(L.Callback):
    """Upload best model to HuggingFace periodically."""
    
    def __init__(
        self,
        hf_repo: str,
        upload_interval: int = 10,
        token: Optional[str] = None,
    ):
        self.hf_repo = hf_repo
        self.upload_interval = upload_interval
        self.token = token
        self.logger = logging.getLogger(__name__)
        
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.upload_interval != 0:
            return
            
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if not best_ckpt or not os.path.exists(best_ckpt):
            self.logger.warning("No best checkpoint found to upload.")
            return
            
        self.logger.info(f"Uploading best checkpoint to HuggingFace: {best_ckpt}")
        try:
            # Generate a simple model card or descriptive string
            model_score = trainer.checkpoint_callback.best_model_score
            card_content = f"""
# Diffusion Classifier OOD Detection

- **Epoch**: {trainer.current_epoch}
- **Best AUROC**: {model_score:.4f} if model_score else 'N/A'
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            push_to_huggingface(
                checkpoint_path=best_ckpt,
                repo_name=self.hf_repo,
                model_card=card_content,
                token=self.token
            )
        except Exception as e:
            self.logger.error(f"Failed to periodic upload to HuggingFace: {e}")
