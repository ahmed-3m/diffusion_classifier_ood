#!/usr/bin/env python3
"""
Evaluate trained binary CDM checkpoints on external OOD datasets.
Produces: external_ood_results.json, raw scores per seed per dataset.
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
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from src.lightning_module import DiffusionClassifierOOD
from src.scoring import diffusion_classifier_score
from src.metrics import compute_all_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def get_transform():
    """Standard transform matching training preprocessing."""
    return transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_grayscale_transform():
    """Transform for grayscale datasets (e.g., FashionMNIST)."""
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_external_ood_datasets(data_dir: str):
    """Load external OOD datasets for evaluation."""
    tfm = get_transform()
    datasets = {}

    # 1. SVHN (Street View House Numbers)
    try:
        svhn = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=tfm
        )
        datasets['svhn'] = svhn
        logger.info(f"  SVHN: {len(svhn)} images loaded")
    except Exception as e:
        logger.warning(f"  SVHN failed: {e}")

    # 2. CIFAR-100
    try:
        cifar100 = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=tfm
        )
        datasets['cifar100'] = cifar100
        logger.info(f"  CIFAR-100: {len(cifar100)} images loaded")
    except Exception as e:
        logger.warning(f"  CIFAR-100 failed: {e}")

    # 3. Textures (DTD)
    try:
        textures = torchvision.datasets.DTD(
            root=data_dir, split='test', download=True, transform=tfm
        )
        datasets['textures'] = textures
        logger.info(f"  Textures (DTD): {len(textures)} images loaded")
    except Exception as e:
        logger.warning(f"  Textures (DTD) failed: {e}")

    # 4. FashionMNIST (as Places365 alternative — much faster to download)
    try:
        fmnist = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True,
            transform=get_grayscale_transform()
        )
        datasets['fashionmnist'] = fmnist
        logger.info(f"  FashionMNIST: {len(fmnist)} images loaded")
    except Exception as e:
        logger.warning(f"  FashionMNIST failed: {e}")

    # 5. Try Places365 (small version)
    try:
        places = torchvision.datasets.Places365(
            root=os.path.join(data_dir, 'places365'),
            split='val', small=True, download=True, transform=tfm
        )
        # Subsample to 10000 for efficiency
        indices = np.random.RandomState(42).choice(len(places), min(10000, len(places)), replace=False)
        places = Subset(places, indices.tolist())
        datasets['places365'] = places
        logger.info(f"  Places365: {len(places)} images loaded")
    except Exception as e:
        logger.warning(f"  Places365 failed (using FashionMNIST as backup): {e}")

    return datasets


def get_cifar10_id_test(data_dir: str, id_class: int = 0):
    """Get CIFAR-10 ID test set (airplane only, ~1000 images)."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    full_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=tfm
    )
    id_indices = [i for i, (_, label) in enumerate(full_test) if label == id_class]
    ood_indices = [i for i, (_, label) in enumerate(full_test) if label != id_class]
    return Subset(full_test, id_indices), Subset(full_test, ood_indices)


def score_images(model, scheduler, images, device, num_trials=50,
                 scoring_method="difference", timestep_mode="mid_focus"):
    """Score a batch of images."""
    images = images.to(device)
    scores, preds = diffusion_classifier_score(
        model.model, scheduler, images,
        num_conditions=2,
        num_trials=num_trials,
        scoring_method=scoring_method,
        timestep_mode=timestep_mode,
    )
    return scores.cpu(), preds.cpu()


def score_dataset_generic(model, scheduler, dataloader, device, num_trials=50,
                          scoring_method="difference", timestep_mode="mid_focus",
                          desc="Scoring"):
    """Score an entire dataset (generic — works with any dataloader)."""
    all_scores = []
    all_preds = []

    for batch in tqdm(dataloader, desc=desc):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        elif isinstance(batch, dict):
            images = batch["images"]
        else:
            images = batch

        scores, preds = score_images(
            model, scheduler, images, device,
            num_trials=num_trials,
            scoring_method=scoring_method,
            timestep_mode=timestep_mode,
        )
        all_scores.append(scores)
        all_preds.append(preds)

    return torch.cat(all_scores), torch.cat(all_preds)


def find_best_checkpoint(seed_dir: str) -> str:
    """Find the best checkpoint in a seed directory."""
    # Look for best- checkpoints
    ckpts = glob(os.path.join(seed_dir, "**", "best-*.ckpt"), recursive=True)
    if ckpts:
        # Return the most recent one
        return max(ckpts, key=os.path.getmtime)

    # Fallback to last.ckpt
    ckpts = glob(os.path.join(seed_dir, "**", "last.ckpt"), recursive=True)
    if ckpts:
        return max(ckpts, key=os.path.getmtime)

    raise FileNotFoundError(f"No checkpoint found in {seed_dir}")


def evaluate_checkpoint(checkpoint_path, device, data_dir, num_trials=50,
                        scoring_method="difference", timestep_mode="mid_focus",
                        batch_size=64, save_prefix=""):
    """Evaluate a single checkpoint on all datasets."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {checkpoint_path}")
    logger.info(f"{'='*60}")

    # Load model
    model = DiffusionClassifierOOD.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    scheduler = model.scheduler

    results = {}

    # 1. Within-CIFAR binary OOD (airplane vs rest)
    logger.info("\n--- Within-CIFAR Binary OOD ---")
    id_test, ood_test = get_cifar10_id_test(data_dir, id_class=0)

    id_loader = DataLoader(id_test, batch_size=batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_test, batch_size=batch_size, shuffle=False, num_workers=4)

    id_scores, _ = score_dataset_generic(
        model, scheduler, id_loader, device,
        num_trials=num_trials, scoring_method=scoring_method,
        timestep_mode=timestep_mode, desc="Scoring ID (airplane)"
    )
    ood_scores, _ = score_dataset_generic(
        model, scheduler, ood_loader, device,
        num_trials=num_trials, scoring_method=scoring_method,
        timestep_mode=timestep_mode, desc="Scoring OOD (CIFAR rest)"
    )

    all_scores = torch.cat([id_scores, ood_scores]).numpy()
    all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    metrics = compute_all_metrics(all_labels, all_scores)
    results['within_cifar'] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
    logger.info(f"  AUROC: {metrics['auroc']:.4f} | FPR95: {metrics['fpr95']:.4f}")

    # Save raw scores
    if save_prefix:
        torch.save(id_scores, f"{save_prefix}_cifar10_id_scores.pt")
        torch.save(ood_scores, f"{save_prefix}_cifar10_ood_scores.pt")

    # 2. External OOD datasets
    logger.info("\nLoading external OOD datasets...")
    external_datasets = load_external_ood_datasets(data_dir)

    for name, dataset in external_datasets.items():
        logger.info(f"\n--- External OOD: {name} ({len(dataset)} images) ---")
        ext_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        ext_scores, _ = score_dataset_generic(
            model, scheduler, ext_loader, device,
            num_trials=num_trials, scoring_method=scoring_method,
            timestep_mode=timestep_mode, desc=f"Scoring {name}"
        )

        # OOD detection: ID (airplane) vs external OOD
        combined_scores = torch.cat([id_scores, ext_scores]).numpy()
        combined_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ext_scores))])
        ext_metrics = compute_all_metrics(combined_labels, combined_scores)
        results[name] = {k: float(v) for k, v in ext_metrics.items() if isinstance(v, (int, float, np.floating))}
        logger.info(f"  AUROC: {ext_metrics['auroc']:.4f} | FPR95: {ext_metrics['fpr95']:.4f}")

        if save_prefix:
            torch.save(ext_scores, f"{save_prefix}_{name}_scores.pt")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on external OOD datasets")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--scoring_method", type=str, default="difference")
    parser.add_argument("--timestep_mode", type=str, default="mid_focus")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    all_results = {}
    seeds = {"seed_42": "seed42", "seed_123": "seed123", "seed_456": "seed456"}

    for seed_name, seed_dir_name in seeds.items():
        seed_dir = os.path.join(args.results_dir, seed_dir_name)
        if not os.path.exists(seed_dir):
            logger.warning(f"Seed directory not found: {seed_dir}, skipping")
            continue

        try:
            ckpt = find_best_checkpoint(seed_dir)
            save_prefix = os.path.join(args.results_dir, "raw_scores", seed_dir_name)
            result = evaluate_checkpoint(
                ckpt, device, args.data_dir,
                num_trials=args.num_trials,
                scoring_method=args.scoring_method,
                timestep_mode=args.timestep_mode,
                batch_size=args.batch_size,
                save_prefix=save_prefix,
            )
            all_results[seed_name] = result
        except FileNotFoundError as e:
            logger.error(f"Checkpoint not found for {seed_name}: {e}")
            continue

    # Compute mean ± std across seeds
    if len(all_results) > 1:
        logger.info("\n" + "="*60)
        logger.info("AGGREGATED RESULTS (mean ± std)")
        logger.info("="*60)

        # Get all dataset names
        dataset_names = set()
        for seed_result in all_results.values():
            dataset_names.update(seed_result.keys())

        aggregated = {}
        for ds_name in dataset_names:
            ds_metrics = {}
            for metric_name in ['auroc', 'fpr95', 'aupr']:
                values = [all_results[s][ds_name][metric_name]
                          for s in all_results if ds_name in all_results[s]
                          and metric_name in all_results[s][ds_name]]
                if values:
                    ds_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                    ds_metrics[f"{metric_name}_std"] = float(np.std(values))

            aggregated[ds_name] = ds_metrics
            if 'auroc_mean' in ds_metrics:
                logger.info(f"  {ds_name}: AUROC={ds_metrics['auroc_mean']:.4f}±{ds_metrics['auroc_std']:.4f}")

        all_results['aggregated'] = aggregated

    # Save results
    output_path = os.path.join(args.results_dir, "external_ood_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
