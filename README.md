# Diffusion Classifier for OOD Detection

Conditional diffusion model for out-of-distribution detection on CIFAR-10.

## Overview

This project implements Algorithm 1 (Diffusion Classifier) from the thesis "Conditional diffusion models as generative classifiers for out-of-distribution detection". The approach uses a binary conditional diffusion model where:

- **c=0**: In-distribution class (airplane by default)
- **c=1**: Out-of-distribution proxy (all other CIFAR-10 classes)

OOD detection is performed by comparing reconstruction errors across conditions.

## Installation

```bash
pip install -r requirements.txt
```

Set up W&B (optional but recommended):
```bash
wandb login
```

## Quick Start

### Training

```bash
# Basic training
python scripts/train.py --experiment_tag baseline

# With custom settings
python scripts/train.py \
    --batch_size 128 \
    --max_epochs 200 \
    --num_trials 15 \
    --experiment_tag high_trials

# Resume from checkpoint
python scripts/train.py --resume --experiment_tag continued
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint_path path/to/best.ckpt
```

## Configuration

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 64 | Training batch size |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--max_epochs` | 200 | Maximum training epochs |
| `--num_trials` | 10 | Scoring trials (Algorithm 1) |
| `--eval_interval` | 10 | Epochs between validation |
| `--id_class` | 0 | CIFAR-10 class to use as ID |

## Project Structure

```
diffusion_classifier_ood/
├── configs/
│   └── default.py          # Dataclass configurations
├── src/
│   ├── model.py            # ConditionalUNet wrapper
│   ├── data.py             # CIFAR-10 data module
│   ├── lightning_module.py # Training/validation logic
│   ├── scoring.py          # Algorithm 1 implementation
│   ├── metrics.py          # OOD detection metrics
│   ├── plotting.py         # Visualization functions
│   └── utils.py            # Utilities and callbacks
├── scripts/
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── requirements.txt
└── README.md
```

## Metrics

The model is evaluated using standard OOD detection metrics:

- **AUROC**: Area Under ROC Curve
- **FPR@95**: False Positive Rate at 95% True Positive Rate
- **AUPR**: Area Under Precision-Recall Curve

## W&B Integration

All training runs log to Weights & Biases:
- Training loss curves
- Validation metrics
- Generated samples
- OOD detection plots

## Citation

```bibtex
@mastersthesis{thesis2025,
  title={Conditional diffusion models as generative classifiers for out-of-distribution detection},
  author={[Ahmed Mohammed]},
  school={Johannes Kepler University Linz},
  year={2025}
}
```

## License

MIT
