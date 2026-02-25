import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import wandb

from src.lightning_module import DiffusionClassifierOOD
from src.data import CIFAR10BinaryDataModule
from src.scoring import score_dataset, compute_per_timestep_errors
from src.metrics import compute_all_metrics, format_metrics_table
from src.plotting import (
    plot_roc_curve,
    plot_precision_recall,
    plot_score_histogram,
    plot_score_violin,
    plot_det_curve,
    plot_fpr_vs_threshold,
    plot_confusion_matrix,
    plot_tsne_embeddings,
    plot_timestep_error,
    plot_extreme_samples,
    plot_generated_samples,
    log_all_plots_to_wandb,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate diffusion classifier for OOD detection")
    
    parser.add_argument("--checkpoint_path", type=str, required=True)
    
    parser.add_argument("--num_trials", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--id_class", type=int, default=0)
    
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--project_name", type=str, default="diffusion-classifier-ood")
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading model from: {args.checkpoint_path}")
    model = DiffusionClassifierOOD.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    data = CIFAR10BinaryDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        id_class=args.id_class,
    )
    data.setup("test")
    
    if args.log_to_wandb:
        wandb.init(
            project=args.project_name,
            name=f"eval_{os.path.basename(args.checkpoint_path)}",
            config=vars(args),
        )
    
    logger.info("Running OOD scoring on test set...")
    scores, predictions, labels = score_dataset(
        model.model,
        model.scheduler,
        data.test_dataloader(),
        num_conditions=2,
        num_trials=args.num_trials,
        device=device,
    )
    
    scores_np = scores.numpy()
    preds_np = predictions.numpy()
    labels_np = labels.numpy()
    
    metrics = compute_all_metrics(labels_np, scores_np, preds_np)
    
    logger.info("\n" + format_metrics_table(metrics))
    
    id_mask = labels_np == 0
    ood_mask = labels_np == 1
    id_scores = scores_np[id_mask]
    ood_scores = scores_np[ood_mask]
    
    plots = {}
    
    plots['roc_curve'] = plot_roc_curve(labels_np, scores_np, metrics['auroc'])
    plots['pr_curve'] = plot_precision_recall(labels_np, scores_np, metrics['aupr'])
    plots['score_histogram'] = plot_score_histogram(id_scores, ood_scores)
    plots['score_violin'] = plot_score_violin(id_scores, ood_scores)
    plots['det_curve'] = plot_det_curve(labels_np, scores_np)
    plots['fpr_threshold'] = plot_fpr_vs_threshold(labels_np, scores_np, metrics['threshold_95'])
    plots['confusion_matrix'] = plot_confusion_matrix(labels_np, preds_np)
    
    if len(scores_np) > 100:
        plots['tsne'] = plot_tsne_embeddings(scores_np, labels_np)
    
    logger.info("Generating samples...")
    samples_c0 = model.sample_images(16, condition=0)
    samples_c1 = model.sample_images(16, condition=1)
    plots['samples_c0'] = plot_generated_samples(samples_c0, "Generated c=0 (ID)")
    plots['samples_c1'] = plot_generated_samples(samples_c1, "Generated c=1 (OOD)")
    
    for name, fig in plots.items():
        path = os.path.join(args.output_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {path}")
        
        if args.log_to_wandb:
            wandb.log({f"eval/{name}": wandb.Image(fig)})
    
    if args.log_to_wandb:
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.finish()
    
    results_path = os.path.join(args.output_dir, "metrics.txt")
    with open(results_path, 'w') as f:
        f.write(format_metrics_table(metrics))
    
    logger.info(f"Evaluation complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
