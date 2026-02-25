# GPU Usage Log

Tracking all GPU allocations across servers during the thesis experiments.

---

## Servers Available

| Server | GPUs | GPU Model | VRAM | Architecture | Tensor Cores |
|--------|------|-----------|------|-------------|--------------|
| student10 | 0 | P40 | 24GB | Pascal | ❌ |
| student10 | 1 | V100 | 16GB | Volta | ✅ |
| student10 | 2 | P40 | 24GB | Pascal | ❌ |
| student10 | 3 | Titan V | 12GB | Volta | ✅ |
| student06 | 0–3 | Titan V ×4 | 12GB each | Volta | ✅ |
| student06 | Topology | GPU0↔GPU1: PHB, GPU2↔GPU3: PHB, cross-pair: SYS |

---

## Allocation Log

### Feb 17–19
| GPU | Server | Job |
|-----|--------|-----|
| 2 | student10 | sep_0.0 (initial slow run, competing processes) |
| 2 | student10 | Various zombie sep runs (killed) |

### Feb 19–20
| GPU | Server | Job |
|-----|--------|-----|
| 1 | student10 | Seed 456 retraining (V100) — SUCCESS |
| 2 | student10 | sep_0.0 (resumed from epoch 19) |
| 3 | student10 | OOD eval (evaluate_external_ood.py) |

### Feb 20–21
| GPU | Server | Job |
|-----|--------|-----|
| 3 | student10 | sep_0.0 full training (200 epochs) — DONE AUROC=0.8025 |

### Feb 21–22
| GPU | Server | Job |
|-----|--------|-----|
| 3 | student10 | sep_0.001 (automatic sequential, after sep_0.0 done) |
| 0 | student06 | sep_0.05 (single GPU) |
| 1 | student06 | sep_0.1 (single GPU) |
| 2 | student06 | sep_0.001 (launched to speed up, but student10 already running it) |

### Feb 22–23
| GPU | Server | Job |
|-----|--------|-----|
| 0 | student06 | sep_0.05 — DONE AUROC=0.9851 |
| 1 | student06 | sep_0.1 — DONE AUROC=0.9667 |
| 3 | student10 | sep_0.001 — DONE AUROC=0.9732 |

### Feb 23–24
| GPU | Server | Job |
|-----|--------|-----|
| 3 | *(32GB server)* | sep_0.01 — DONE AUROC=0.9869 |
| 3 | *(32GB server)* | sep_0.02 — DONE AUROC=0.9786 |

---

## DDP Attempts & Failures

### Feb 22 13:06 — DDP Attempt on student06
- **Config**: `CUDA_VISIBLE_DEVICES=0,1` for sep_0.05, `2,3` for sep_0.1
- **Failure**: NCCL timeout after 30 min
  ```
  [Rank 1] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=1, OpType=BROADCAST) ran for 1800007ms
  ```
- **Root cause**: OOD sanity check (15 trials × 2 batches) took 30+ min on rank 1;
  rank 0 finished and broadcast timed out waiting for rank 1
- **Fix applied**: Added `num_sanity_val_steps=0` to `L.Trainer()` in `train.py`
- **Lesson**: With slow per-sample scoring, sanity check blocks DDP sync

### GPU Topology Note
- student06 is dual-CPU (NUMA 0: GPU0+GPU1, NUMA 1: GPU2+GPU3)
- PHB pairs (same NUMA) are safe for DDP
- SYS pairs (cross-NUMA) risk NCCL timeouts
- 4-GPU DDP (0+1+2+3) crosses NUMA boundary — not recommended

---

## Common Issues & Fixes

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| 0.35 it/s training speed | Zombie processes on same GPU | `pkill -9 -u mohammed -f "scripts/train.py"` |
| NCCL timeout in DDP | Sanity check OOD scoring too slow | `num_sanity_val_steps=0` in Trainer |
| P40 slow training | No Tensor Cores → `16-mixed` gives no benefit | Use Titan V or V100 instead |
| OOM on GPU | Competing student processes | Check `nvidia-smi`, use different GPU |
| `tqdm` bar missing in log file | Non-TTY redirect strips formatting | Use tmux foreground, not `> file &` |
| `expandable_segments` warning | P40/old driver doesn't support it | Harmless warning, ignore |
