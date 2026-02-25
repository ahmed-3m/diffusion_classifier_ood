#!/usr/bin/env python3
"""
Evaluate models trained with different separation loss weights (Experiment Set 6).
Produces: separation_loss_results.json
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lightning_module import DiffusionClassifierOOD
from src.metrics import compute_all_metrics
from scripts.evaluate_external_ood import (
    find_best_checkpoint,
    get_cifar10_id_test,
    score_dataset_generic,
    load_external_ood_datasets,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Evaluate separation loss ablation models")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We expect directories like 'results/sep_loss_ablation/sep_WEIGHT'
    # Actually, train.py saves to `args.output_dir/run_name`.
    # And run_all_experiments.sh sets --output_dir "${RESULTS_DIR}/sep_loss_ablation"
    # So we look for subdirectories there.
    
    base_dir = os.path.join(args.results_dir, "sep_loss_ablation")
    if not os.path.exists(base_dir):
        logger.warning(f"Abation directory not found: {base_dir}")
        return

    # Weights to evaluate
    weights = [0.0, 0.001, 0.01, 0.05, 0.1]
    
    # Data loaders
    id_test, ood_test = get_cifar10_id_test(args.data_dir)
    id_loader = DataLoader(id_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # SVHN check
    try:
        import torchvision.transforms as transforms
        import torchvision
        tfm = transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32),
            transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        svhn = torchvision.datasets.SVHN(root=args.data_dir, split='test', download=True, transform=tfm)
        svhn_loader = DataLoader(svhn, batch_size=args.batch_size, shuffle=False, num_workers=4)
        has_svhn = True
    except:
        has_svhn = False
        svhn_loader = None

    results = {"weights": weights, "within_cifar": {}, "svhn": {}}

    # Also include the baseline (0.01) from Seed 42 main run if available
    # But usually we just re-evaluate it here if it's in the list
    # The master script runs 0.0, 0.001, 0.05, 0.1.
    # 0.01 is the default trained in Experiment 1 (Seed 42).
    # So for 0.01, we should point to seed42 results.
    
    for w in weights:
        logger.info(f"\nProcessing weight: {w}...")
        
        if w == 0.01:
            # Check seed42 directory first
            ckpt_dir = os.path.join(args.results_dir, "seed42")
        else:
            # Look for directory starting with sep_{w}
            # The run script creates directories like: results/sep_loss_ablation/2025..._sep_0.0/
            # We need to find the specific directory.
            candidates = []
            for d in os.listdir(base_dir):
                if f"sep_{w}" in d:
                    candidates.append(os.path.join(base_dir, d))
            
            if not candidates:
                logger.warning(f"  No directory found for weight {w}")
                continue
                
            ckpt_dir = max(candidates, key=os.path.getmtime)
            
        try:
            ckpt_path = find_best_checkpoint(ckpt_dir)
            logger.info(f"  Checkpoint: {ckpt_path}")
            
            model = DiffusionClassifierOOD.load_from_checkpoint(ckpt_path)
            model.eval()
            model = model.to(device)
            scheduler = model.scheduler
            
            # Score
            id_scores, _ = score_dataset_generic(
                model, scheduler, id_loader, device, num_trials=15, desc=f"ID w={w}"
            )
            ood_scores, _ = score_dataset_generic(
                model, scheduler, ood_loader, device, num_trials=15, desc=f"OOD w={w}"
            )
            
            all_scores = torch.cat([id_scores, ood_scores]).numpy()
            all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
            metrics = compute_all_metrics(all_labels, all_scores)
            
            results["within_cifar"][str(w)] = {
                "auroc": float(metrics['auroc']),
                "fpr95": float(metrics['fpr95']),
            }
            logger.info(f"    CIFAR: AUROC={metrics['auroc']:.4f}")
            
            if has_svhn:
                svhn_scores, _ = score_dataset_generic(
                    model, scheduler, svhn_loader, device, num_trials=15, desc=f"SVHN w={w}"
                )
                combined_scores = torch.cat([id_scores, svhn_scores]).numpy()
                combined_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(svhn_scores))])
                svhn_metrics = compute_all_metrics(combined_labels, combined_scores)
                
                results["svhn"][str(w)] = {
                    "auroc": float(svhn_metrics['auroc']),
                    "fpr95": float(svhn_metrics['fpr95']),
                }
                logger.info(f"    SVHN:  AUROC={svhn_metrics['auroc']:.4f}")
                
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"  Failed for w={w}: {e}")
            continue

    output_path = os.path.join(args.results_dir, "separation_loss_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
