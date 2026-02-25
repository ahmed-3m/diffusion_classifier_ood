import os
import sys
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb

from configs.default import Config, ModelConfig, TrainingConfig, DataConfig, EvalConfig, LoggingConfig
from src.lightning_module import DiffusionClassifierOOD
from src.data import CIFAR10BinaryDataModule
from src.utils import (
    setup_logging,
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    MemoryCleanupCallback,
    SampleVisualizationCallback,
    push_to_huggingface,
)
from src.model import generate_model_card

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion classifier for OOD detection")
    
    parser.add_argument("--experiment_tag", type=str, default="run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint file or directory containing checkpoints")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--id_class", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="./data")
    
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--project_name", type=str, default="diffusion-classifier-ood")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    
    parser.add_argument("--upload_to_hf", action="store_true")
    parser.add_argument("--hf_repo", type=str, default="")
    parser.add_argument("--upload_interval", type=int, default=10, help="Epoch interval for uploading best model to HF")
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = args.wandb_mode
    
    run_name = f"{datetime.now():%Y-%m-%d/%H-%M-%S}_{args.experiment_tag}"
    output_dir = os.path.join(args.output_dir, run_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Experiment: {run_name}")
    
    model = DiffusionClassifierOOD(
        num_train_timesteps=1000,
        num_class_embeds=2,
        num_trials=args.num_trials,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_epochs=5,
    )
    
    data = CIFAR10BinaryDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        id_class=args.id_class,
    )
    
    wandb_logger = WandbLogger(
        save_dir=output_dir,
        project=args.project_name,
        name=run_name,
        log_model=False,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/auroc",
        dirpath=output_dir,
        filename="best-{epoch:02d}-{val/auroc:.4f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val/auroc",
        patience=30,
        mode="max",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    sample_callback = SampleVisualizationCallback(
        every_n_epochs=args.eval_interval,
        num_samples=8,
    )
    
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        log_every_n_steps=50,
        default_root_dir=output_dir,
        devices="auto",
        accelerator="auto",
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            MemoryCleanupCallback(),
            sample_callback,
        ],
        enable_progress_bar=True,
        check_val_every_n_epoch=args.eval_interval,
    )

    if args.upload_to_hf and args.hf_repo:
        upload_callback = HuggingFaceUploadCallback(
            hf_repo=args.hf_repo,
            upload_interval=args.upload_interval,
        )
        trainer.callbacks.append(upload_callback)
    
    resume_ckpt = None
    if args.checkpoint_path:
        if os.path.isdir(args.checkpoint_path):
            logger.info(f"Searching for latest checkpoint in: {args.checkpoint_path}")
            resume_ckpt = find_latest_checkpoint(args.checkpoint_path)
            if not resume_ckpt:
                logger.warning(f"No checkpoint found in {args.checkpoint_path}, starting fresh")
        else:
            resume_ckpt = args.checkpoint_path
            
    elif args.resume:
        resume_ckpt = find_latest_checkpoint(args.output_dir)
    
    if resume_ckpt:
        logger.info(f"Resuming from: {resume_ckpt}")
    else:
        if args.resume or args.checkpoint_path:
            logger.warning("Resume requested but no checkpoint found. Starting fresh.")
    
    logger.info("=" * 60)
    logger.info(" DIFFUSION CLASSIFIER FOR OOD DETECTION")
    logger.info(" Training started")
    logger.info("=" * 60)
    
    torch.cuda.empty_cache()
    trainer.fit(model, datamodule=data, ckpt_path=resume_ckpt)
    
    cleaned = cleanup_old_checkpoints(output_dir, keep_last=1, keep_best=True)
    logger.info(f"Cleaned up {cleaned} old checkpoints")
    
    if args.upload_to_hf and args.hf_repo:
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt:
            try:
                model_card = generate_model_card(
                    model.hparams,
                    {"auroc": checkpoint_callback.best_model_score or 0},
                    f"{args.max_epochs} epochs",
                )
                push_to_huggingface(best_ckpt, args.hf_repo, model_card)
            except Exception as e:
                logger.error(f"Failed to upload to HuggingFace: {e}")
    
    logger.info("=" * 60)
    logger.info(" TRAINING COMPLETE")
    logger.info(f" Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f" Best AUROC: {checkpoint_callback.best_model_score:.4f}")
    logger.info("=" * 60)
    
    wandb.finish()


if __name__ == "__main__":
    main()
