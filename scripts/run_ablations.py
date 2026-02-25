#!/usr/bin/env python3
"""
Run ablation studies (Experiments 3-5):
  3. K (num_trials) ablation
  4. Timestep strategy ablation
  5. Scoring method ablation

Uses the best seed42 checkpoint for all ablations.
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from src.lightning_module import DiffusionClassifierOOD
from src.scoring import diffusion_classifier_score, compute_per_timestep_errors
from src.metrics import compute_all_metrics
from scripts.evaluate_external_ood import (
    find_best_checkpoint,
    get_cifar10_id_test,
    score_dataset_generic,
    load_external_ood_datasets,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def run_k_ablation(model, scheduler, device, data_dir, results_dir, batch_size=64):
    """Experiment 3: How many trials K are needed?"""
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: K (num_trials) Ablation")
    logger.info("="*60)

    K_values = [1, 5, 10, 25, 50, 100]
    results = {"within_cifar": {}, "svhn": {}}

    id_test, ood_test = get_cifar10_id_test(data_dir)
    id_loader = DataLoader(id_test, batch_size=batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load SVHN
    tfm = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    try:
        svhn = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=tfm)
        svhn_loader = DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=4)
        has_svhn = True
    except:
        has_svhn = False
        logger.warning("SVHN not available, skipping")

    for K in K_values:
        logger.info(f"\n  K={K}...")

        # Within-CIFAR
        start_time = time.time()
        id_scores, _ = score_dataset_generic(
            model, scheduler, id_loader, device,
            num_trials=K, desc=f"ID K={K}"
        )
        ood_scores, _ = score_dataset_generic(
            model, scheduler, ood_loader, device,
            num_trials=K, desc=f"OOD K={K}"
        )
        elapsed = time.time() - start_time

        all_scores = torch.cat([id_scores, ood_scores]).numpy()
        all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        metrics = compute_all_metrics(all_labels, all_scores)

        results["within_cifar"][f"K_{K}"] = {
            "auroc": float(metrics['auroc']),
            "fpr95": float(metrics['fpr95']),
            "aupr": float(metrics['aupr']),
            "time_seconds": elapsed,
            "time_per_sample": elapsed / len(all_scores),
        }
        logger.info(f"    CIFAR: AUROC={metrics['auroc']:.4f} FPR95={metrics['fpr95']:.4f} time={elapsed:.1f}s")

        # SVHN
        if has_svhn:
            start_time = time.time()
            svhn_scores, _ = score_dataset_generic(
                model, scheduler, svhn_loader, device,
                num_trials=K, desc=f"SVHN K={K}"
            )
            elapsed_svhn = time.time() - start_time

            combined_scores = torch.cat([id_scores, svhn_scores]).numpy()
            combined_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(svhn_scores))])
            svhn_metrics = compute_all_metrics(combined_labels, combined_scores)

            results["svhn"][f"K_{K}"] = {
                "auroc": float(svhn_metrics['auroc']),
                "fpr95": float(svhn_metrics['fpr95']),
                "aupr": float(svhn_metrics['aupr']),
                "time_seconds": elapsed_svhn,
            }
            logger.info(f"    SVHN:  AUROC={svhn_metrics['auroc']:.4f} FPR95={svhn_metrics['fpr95']:.4f}")

    output_path = os.path.join(results_dir, "k_ablation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nK ablation results saved to: {output_path}")

    return results


def run_timestep_ablation(model, scheduler, device, data_dir, results_dir, batch_size=64):
    """Experiment 4: Timestep sampling strategy comparison."""
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 4: Timestep Strategy Ablation")
    logger.info("="*60)

    strategies = ["uniform", "mid_focus", "stratified"]
    results = {"within_cifar": {}, "svhn": {}}

    id_test, ood_test = get_cifar10_id_test(data_dir)
    id_loader = DataLoader(id_test, batch_size=batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load SVHN
    tfm = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    try:
        svhn = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=tfm)
        svhn_loader = DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=4)
        has_svhn = True
    except:
        has_svhn = False

    for strategy in strategies:
        logger.info(f"\n  Strategy: {strategy}...")

        id_scores, _ = score_dataset_generic(
            model, scheduler, id_loader, device,
            num_trials=50, timestep_mode=strategy, desc=f"ID {strategy}"
        )
        ood_scores, _ = score_dataset_generic(
            model, scheduler, ood_loader, device,
            num_trials=50, timestep_mode=strategy, desc=f"OOD {strategy}"
        )

        all_scores = torch.cat([id_scores, ood_scores]).numpy()
        all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        metrics = compute_all_metrics(all_labels, all_scores)

        results["within_cifar"][strategy] = {
            "auroc": float(metrics['auroc']),
            "fpr95": float(metrics['fpr95']),
            "aupr": float(metrics['aupr']),
        }
        logger.info(f"    CIFAR: AUROC={metrics['auroc']:.4f} FPR95={metrics['fpr95']:.4f}")

        if has_svhn:
            svhn_scores, _ = score_dataset_generic(
                model, scheduler, svhn_loader, device,
                num_trials=50, timestep_mode=strategy, desc=f"SVHN {strategy}"
            )
            combined_scores = torch.cat([id_scores, svhn_scores]).numpy()
            combined_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(svhn_scores))])
            svhn_metrics = compute_all_metrics(combined_labels, combined_scores)
            results["svhn"][strategy] = {
                "auroc": float(svhn_metrics['auroc']),
                "fpr95": float(svhn_metrics['fpr95']),
                "aupr": float(svhn_metrics['aupr']),
            }
            logger.info(f"    SVHN:  AUROC={svhn_metrics['auroc']:.4f}")

    # Also compute per-timestep error analysis for the figure
    logger.info("\n  Computing per-timestep errors for figure...")
    id_loader_small = DataLoader(id_test, batch_size=min(100, len(id_test)), shuffle=False, num_workers=4)
    ood_loader_small = DataLoader(ood_test, batch_size=min(100, len(ood_test)), shuffle=False, num_workers=4)

    id_batch = next(iter(id_loader_small))
    ood_batch = next(iter(ood_loader_small))
    id_images = id_batch[0].to(device) if isinstance(id_batch, (list, tuple)) else id_batch["images"].to(device)
    ood_images = ood_batch[0].to(device) if isinstance(ood_batch, (list, tuple)) else ood_batch["images"].to(device)

    timesteps_to_eval = list(range(50, 951, 50))

    per_timestep = {"timesteps": timesteps_to_eval, "id": {}, "ood": {}}
    for c in [0, 1]:
        id_errors = compute_per_timestep_errors(model.model, scheduler, id_images, timesteps_to_eval, condition=c)
        ood_errors = compute_per_timestep_errors(model.model, scheduler, ood_images, timesteps_to_eval, condition=c)
        for t in timesteps_to_eval:
            per_timestep["id"][f"c{c}_t{t}"] = {"mean": id_errors[t]['mean'], "std": id_errors[t]['std']}
            per_timestep["ood"][f"c{c}_t{t}"] = {"mean": ood_errors[t]['mean'], "std": ood_errors[t]['std']}

    results["per_timestep"] = per_timestep

    output_path = os.path.join(results_dir, "timestep_strategy_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nTimestep strategy results saved to: {output_path}")

    return results


def run_scoring_method_ablation(model, scheduler, device, data_dir, results_dir, batch_size=64):
    """Experiment 5: Scoring method comparison."""
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 5: Scoring Method Ablation")
    logger.info("="*60)

    methods = ["difference", "ratio", "id_error"]
    results = {"within_cifar": {}, "svhn": {}}

    id_test, ood_test = get_cifar10_id_test(data_dir)
    id_loader = DataLoader(id_test, batch_size=batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load SVHN
    tfm = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    try:
        svhn = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=tfm)
        svhn_loader = DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=4)
        has_svhn = True
    except:
        has_svhn = False

    for method in methods:
        logger.info(f"\n  Method: {method}...")

        id_scores, _ = score_dataset_generic(
            model, scheduler, id_loader, device,
            num_trials=50, scoring_method=method, desc=f"ID {method}"
        )
        ood_scores, _ = score_dataset_generic(
            model, scheduler, ood_loader, device,
            num_trials=50, scoring_method=method, desc=f"OOD {method}"
        )

        all_scores = torch.cat([id_scores, ood_scores]).numpy()
        all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        metrics = compute_all_metrics(all_labels, all_scores)

        results["within_cifar"][method] = {
            "auroc": float(metrics['auroc']),
            "fpr95": float(metrics['fpr95']),
            "aupr": float(metrics['aupr']),
        }
        logger.info(f"    CIFAR: AUROC={metrics['auroc']:.4f} FPR95={metrics['fpr95']:.4f}")

        if has_svhn:
            svhn_scores, _ = score_dataset_generic(
                model, scheduler, svhn_loader, device,
                num_trials=50, scoring_method=method, desc=f"SVHN {method}"
            )
            combined_scores = torch.cat([id_scores, svhn_scores]).numpy()
            combined_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(svhn_scores))])
            svhn_metrics = compute_all_metrics(combined_labels, combined_scores)
            results["svhn"][method] = {
                "auroc": float(svhn_metrics['auroc']),
                "fpr95": float(svhn_metrics['fpr95']),
                "aupr": float(svhn_metrics['aupr']),
            }
            logger.info(f"    SVHN:  AUROC={svhn_metrics['auroc']:.4f}")

    output_path = os.path.join(results_dir, "scoring_method_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nScoring method results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies (Exp 3-5)")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find the best seed42 checkpoint
    seed42_dir = os.path.join(args.results_dir, "seed42")
    try:
        ckpt_path = find_best_checkpoint(seed42_dir)
    except FileNotFoundError:
        # Fallback: try the existing best checkpoint
        ckpt_path = "outputs/2025-12-28_16-36-49_sep_loss_0.01/best-epoch=19-val/auroc=0.9814.ckpt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("No checkpoint found for ablation studies")

    logger.info(f"Using checkpoint: {ckpt_path}")

    model = DiffusionClassifierOOD.load_from_checkpoint(ckpt_path)
    model.eval()
    model = model.to(device)
    scheduler = model.scheduler

    # Run all 3 ablation studies
    run_k_ablation(model, scheduler, device, args.data_dir, args.results_dir, args.batch_size)
    run_timestep_ablation(model, scheduler, device, args.data_dir, args.results_dir, args.batch_size)
    run_scoring_method_ablation(model, scheduler, device, args.data_dir, args.results_dir, args.batch_size)

    logger.info("\n\nAll ablation studies complete!")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
