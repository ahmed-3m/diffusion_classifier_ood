# Binary Diffusion Classifier for OOD Detection — Documentation

> **Master's Thesis Project** | Mohammed | February 2026

This directory documents all experiments, results, and design decisions for the
**Binary Conditional Diffusion Model (CDM)** for Out-of-Distribution (OOD) detection.

---

## 📁 Documentation Index

| File | Description |
|------|-------------|
| [experiments.md](experiments.md) | Full log of all experiments run (training, evaluation, ablations) |
| [results_summary.md](results_summary.md) | Final results table — all metrics in one place |
| [ablation_study.md](ablation_study.md) | Separation loss ablation study in detail |
| [setup.md](setup.md) | Environment setup & how to reproduce all experiments |
| [gpu_log.md](gpu_log.md) | GPU usage log across servers (student06, student07, student10) |

---

## 🎯 Project Summary

The project trains a **conditional UNet diffusion model** in a binary classification
setting (ID vs. OOD) on CIFAR-10. At inference, OOD detection is performed by
computing the difference in diffusion reconstruction error between the two class
conditions — a method called the **Diffusion Classifier Score**.

### Key Contribution
A novel **Separation Loss** term added to the training objective encourages the model
to produce distinctly different reconstruction errors for ID vs. OOD samples,
directly optimising the OOD detection signal.

### Best Result
- **AUROC: 0.9887** (Seed 456, val set)
- **Mean AUROC: 0.9818 ± 0.0049** across 3 seeds

---

## 📊 Final Figures

All figures are in [`results/figures/`](../results/figures/):

| Figure | Description |
|--------|-------------|
| `separation_loss_ablation_final.png` | ⭐ Main ablation — 6 λ weights |
| `score_distributions_all.png` | OOD score distributions per seed |
| `k_ablation.png` | Effect of number of MC trials K |
| `timestep_strategy_comparison.png` | Timestep sampling strategies |
| `scoring_method_comparison.png` | Difference vs ratio scoring |
| `calibration_curves.png` | Calibration of OOD scores |
| `confusion_matrix.png` | Confusion matrix on test set |

---

## 📐 LaTeX Tables

All tables are in [`results/latex_tables/`](../results/latex_tables/):

- `main_results_table.tex` — Mean/std AUROC across seeds
- `separation_loss_table.tex` — Ablation table (6 weights)
- `k_ablation_table.tex` — K sensitivity table
- `scoring_method_table.tex` — Scoring method comparison
- `timestep_strategy_table.tex` — Timestep strategy comparison
