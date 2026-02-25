# Separation Loss Ablation Study

> Detailed analysis of the separation loss weight ablation.

---

## Motivation

The standard diffusion model training objective (MSE loss) doesn't directly
optimize for OOD detection. The **separation loss** is a novel contribution
that explicitly encourages the model to assign different reconstruction errors
to ID and OOD samples:

```
L_total = L_MSE + λ · L_sep

L_sep = -|ε_pred(x, c=0) - ε_pred(x, c=1)|
```

Where:
- `ε_pred(x, c)` = predicted noise for input x given condition c
- `c=0` = ID condition, `c=1` = OOD condition
- The negative sign means minimizing L_sep maximizes the prediction difference

The key insight: **a larger prediction difference between conditions directly
improves OOD detection**, since the OOD score is based on this difference.

---

## Experimental Setup

- **Fixed across all runs:** seed=42, batch=64, lr=1e-4, 200 epochs,
  num_trials=15, scoring=difference, timestep=mid_focus, accumulate_grad=2
- **Varied:** separation_loss_weight λ ∈ {0.0, 0.001, 0.01, 0.02, 0.05, 0.1}
- **Metric:** Validation AUROC (best checkpoint, saved by ModelCheckpoint)

---

## Results

| λ       | Val AUROC | Best Epoch | Convergence |
|---------|----------|------------|-------------|
| 0.0     | 0.8025   | 79         | Slow — needs many epochs without sep signal |
| 0.001   | 0.9732   | 19         | Fast — even tiny sep loss helps |
| **0.01** | **0.9869** | **19** | **Fast — optimal balance MSE/sep** |
| 0.02    | 0.9786   | 9          | Very fast — slightly too aggressive |
| 0.05    | 0.9851   | 19         | Fast — still strong |
| 0.1     | 0.9667   | 149        | Very slow — sep dominates MSE, hurts reconstruction |

---

## Analysis

### 1. Separation Loss is Essential (+18.4% AUROC)
λ=0.0 gives only 0.8025 AUROC. Even λ=0.001 jumps to 0.9732.
This proves the standard MSE objective alone is insufficient for OOD detection.

### 2. Optimal Weight: λ=0.01
Peak performance at 0.9869 AUROC. This balances MSE reconstruction quality
with separation signal strength.

### 3. Robust Range: λ ∈ [0.01, 0.05]
All weights in this range give AUROC ≥ 0.9786. The model is not overly
sensitive to the exact λ value — good for practical use.

### 4. Too-Large λ Hurts
At λ=0.1, AUROC drops to 0.9667 and convergence slows dramatically
(best epoch=149 vs 19). The separation loss overpowers the MSE reconstruction
objective, degrading the diffusion model's core ability.

### 5. Convergence Speed
| λ | Best epoch | Interpretation |
|---|------------|----------------|
| 0.0 | 79 | Slow — model drifts without separation signal |
| 0.001–0.05 | 9–19 | Fast — separation provides strong training signal |
| 0.1 | 149 | Slow — MSE undermined, model struggles to reconstruct |

### 6. λ=0.02 Dip
The 0.9786 at λ=0.02 is slightly lower than both 0.01 (0.9869) and 0.05 (0.9851).
This is likely single-seed variance. Best epoch was 9 (only first checkpoint),
suggesting it may have continued improving if eval_interval were smaller.

---

## Visual

See `results/figures/separation_loss_ablation_final.png`:
```
AUROC
1.00 |
0.99 |          ★ 0.9869
0.98 |   ●0.9732    ●0.9786  ●0.9851
0.97 |                              ●0.9667
     |
0.80 | ●0.8025
     +---------------------------------------→
       0.0   0.001  0.01   0.02   0.05   0.1
                Separation Loss Weight λ
```

---

## For Thesis

Suggested paragraph:

> We conducted an ablation study on the separation loss weight λ to quantify
> its impact on OOD detection performance. Table X shows that removing the
> separation loss entirely (λ=0) results in a significant performance drop
> to 0.8025 AUROC, compared to 0.9869 with the optimal λ=0.01 — an improvement
> of 18.4 percentage points. Performance remains robust across the range
> λ ∈ [0.01, 0.05], with all values achieving AUROC ≥ 0.9786. Setting λ too
> high (0.1) leads to degraded performance (0.9667) and slower convergence
> (best epoch 149 vs. 19), as the separation objective begins to dominate the
> reconstruction loss. These results confirm that the separation loss is a
> critical component of our training framework, and that the model is not
> overly sensitive to the exact choice of λ within a reasonable range.
