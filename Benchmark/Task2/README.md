# Task 2: Outfit Composition Baselines

This directory contains the baseline implementations for **Task 2 (Outfit Composition)** of the FashionStylist benchmark.

## Baselines

| Model | Source | Description |
|-------|--------|-------------|
| **POG** | Our implementation (based on [Wen et al., 2019](https://dl.acm.org/doi/10.1145/3292500.3330652)) | Transformer-based multimodal outfit model using FashionCLIP features with FITB evaluation |
| **CLHE** | [Official repo](https://github.com/Xiaohao-Liu/CLHE) | Contrastive Learning with Hard-negative Exploitation |
| **CIRP** | [Official repo](https://github.com/HappyPointer/CIRP) | Composed Image Retrieval Pipeline |
| **DiFashion** | [Official repo](https://github.com/YiyanXu/DiFashion) | Diffusion-based fashion outfit generation |
> **Note:** POG is our own implementation following the POG paper. The other three baselines (CLHE, DiFashion, CIRP) directly use their official open-source code — please refer to their respective repositories for setup and usage instructions.



## POG Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training & Evaluation

```bash
# Run with both text modes (title and title_attrs)
python POG.py

# Run with a specific text mode
python POG.py --text-modes title_attrs

# Customize training
python POG.py --epochs 50 --batch-size 64 --fusion-mode concat --lr 2e-4

# Feature extraction only (no training)
python POG.py --prepare-only
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--text-modes` | `title title_attrs` | Text representation modes to run |
| `--fusion-mode` | `concat` | Multimodal fusion: `concat`, `mean`, `image_only`, `text_only` |
| `--data-root` | `data` | Path to split CSV directory |
| `--source-data-root` | `sourceData` | Path to source data directory |
| `--epochs` | `30` | Training epochs |
| `--batch-size` | `32` | Training batch size |
| `--embed-dim` | `256` | Model embedding dimension |
| `--n-layers` | `4` | Transformer encoder layers |
| `--n-heads` | `8` | Attention heads |
| `--prepare-only` | `false` | Only extract and cache features |

### Outputs

```
outputs/
├── title/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── alignment_report.json
│   └── text_samples.json
├── title_attrs/
│   └── ...
└── summary.json
```

### Evaluation Metrics

- **Recall@K** (K = 1, 5, 10, 20, 50) — FITB global retrieval
- **NDCG@K** — Normalized Discounted Cumulative Gain
- Model selection is based on **Recall@10**
