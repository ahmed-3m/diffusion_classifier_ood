# Dataset Description

> CIFAR-10 Binary Out-of-Distribution Detection Setup

---

## Source Dataset

**CIFAR-10** (Krizhevsky, 2009)
- 60,000 colour images, 32×32 pixels, 3 channels (RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Standard split: 50,000 train / 10,000 test

---

## Binary OOD Split

The dataset is converted to a **binary classification** task:

| Label | Meaning | CIFAR-10 classes | Purpose |
|-------|---------|-----------------|---------|
| **c=0** (ID) | In-Distribution | airplane (class 0) | 1 class |
| **c=1** (OOD) | Out-of-Distribution proxy | all other 9 classes | 9 classes |

### Why airplane?
Airplane (class 0) was chosen as the ID class. This creates a challenging
1-vs-9 split where the model must learn what "normal" looks like from only
one semantic category.

---

## Data Splits & Sizes

### Training Set (from CIFAR-10 train, 50k total)
| Subset | Raw samples | After balancing |
|--------|-------------|-----------------|
| ID (c=0, airplane) | 5,000 | 44,992 (oversampled ×9) |
| OOD proxy (c=1, other 9 classes) | 45,000 | 44,992 (undersampled to match) |
| **Total balanced training** | — | **89,984** |

> The `BalancedBinaryDataset` class ensures 50/50 class balance by oversampling
> the smaller class (ID) to match the larger (OOD). This prevents the model
> from trivially predicting "OOD" for everything.

### Validation Set (from CIFAR-10 test, 10k total)
| Subset | Samples |
|--------|---------|
| ID (airplane) | 1,000 |
| OOD (other 9 classes) | 9,000 |
| **Total** | **10,000** |

> Validation uses weighted sampling to address the 1:9 class imbalance,
> ensuring balanced evaluation of both ID and OOD detection.

---

## Data Augmentation

### Training
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # scale to [-1, 1]
])
```

### Validation / Test
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # scale to [-1, 1]
])
```

> Note: Normalization to [-1, 1] (not ImageNet mean/std) is standard for
> diffusion models, as the denoising process operates in this range.

---

## External OOD Datasets (for generalization testing)

The model is also evaluated on completely unseen datasets:

| Dataset | Samples | Image Size | Description |
|---------|---------|------------|-------------|
| **SVHN** | 26,032 (test) | 32×32 | Street View House Numbers |
| **CIFAR-100** | 10,000 (test) | 32×32 | 100 fine-grained classes |

> ⚠️ External OOD evaluation results are currently incomplete
> (`results/external_ood_results.json` is empty). This needs to be re-run.

---

## DataLoader Configuration

| Parameter | Training | Validation |
|-----------|---------|------------|
| batch_size | 64 | 128 |
| shuffle | Yes | No (weighted sampler) |
| num_workers | 4 (default) / 8 (32GB servers) | 4 |
| pin_memory | Yes | Yes |
| drop_last | Yes | No |
| persistent_workers | Yes (if num_workers > 0) | Yes |

---

## Key Design Decisions

1. **Binary framing**: Reduces 10-class problem to 2-class. The diffusion model
   learns to reconstruct "airplane" (c=0) and "not airplane" (c=1), then uses
   reconstruction error differences for OOD detection.

2. **Balanced training**: Without balancing, the model would see 90% OOD samples
   and learn to reconstruct "not airplane" well, defeating the purpose.

3. **Single ID class**: Using one class as ID is the hardest setting. Some OOD
   detection papers use 6 ID classes and 4 OOD. Our 1-vs-9 setting is more
   challenging and demonstrates the model's ability to capture fine-grained
   distribution characteristics.

4. **No test set contamination**: Validation uses CIFAR-10's official test split.
   External OOD datasets (SVHN, CIFAR-100) are completely unseen during training.
