# CORE: Session-Based Recommendation within Consistent Representation Space

> **Reproducing and extending the SIGIR 2022 paper** — *"CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space"* (Hou et al.)

---

## The Core Problem

### Representation Inconsistency in Session-Based Recommendation

Traditional session-based recommenders (GRU4Rec, SASRec, SR-GNN, NARM) follow an **encoder-decoder** architecture:

```
Item Embeddings → Non-linear Encoder (RNN/Transformer/GNN) → Session Embedding → Dot Product → Scores
```

**The geometric flaw:** Non-linear encoders (RNNs, Transformers, GNNs) transform item embeddings through layers of non-linear activations and weight matrices. The resulting **session embedding lives in a different vector space** than the item embeddings. When the decoder computes dot products between the session vector and candidate item vectors, it's measuring similarity between vectors from **incompatible coordinate systems**.

This **representation inconsistency** acts as a fundamental bottleneck — the model cannot faithfully express "this session is similar to these items" because the session vector has been geometrically deformed away from the item space.

---

## The CORE Solution

### Representation Consistency by Design

CORE enforces a **hard constraint**: the session embedding must be a **linear combination of clicked item embeddings**.

```
session_embedding = Σ(αᵢ × item_embeddingᵢ)
```

Since a linear combination of vectors from a space **stays in that same space**, the session representation is **guaranteed** to share the exact same embedding space as items. The dot product (or cosine similarity) becomes geometrically valid.

### Two Variants

| Variant | Weight Computation | Complexity |
|---------|-------------------|------------|
| **CORE-ave** | Uniform: αᵢ = 1/L | O(1) — mean pooling |
| **CORE-trm** | Transformer attention | O(L²) — learned weights |

Both use the **same item embedding table** for encoding and decoding — true parameter sharing.

### Robust Distance Measuring

In the unified space, dot products are sensitive to vector norms (popularity bias). CORE uses:
- **Cosine similarity** (L2-normalized vectors)
- **Learnable temperature** τ (sharpens/softens distribution)
- **Controllable margin** in InfoNCE loss
- **Item dropout** during training (prevents overfitting)

---

## What We Built

### Complete Reproduction Pipeline

| Milestone | Component | Key Details |
|-----------|-----------|-------------|
| **A** | Data Pipeline | Strict temporal split (80/10/10 by session start date), freq≥5 vocab, prefix-target pairs |
| **B** | Evaluation | Recall@20, MRR@20, grouped by prefix length & item popularity, synthetic unit tests |
| **C** | Baselines | POP (0.89%), Transition (34.15%), Session-POP (21.32%) |
| **D** | CORE-ave | Mean pooling + cosine + InfoNCE + margin |
| **E** | CORE-trm | Causal Transformer → attention weights → weighted sum of **original** embeddings |
| **F** | API + Frontend | FastAPI `/recommend` endpoint + glassmorphism UI |

### Key Implementation Decisions

1. **Temporal integrity**: Split by session *start date*, not random — prevents future leakage
2. **Vocabulary from train only**: Items with freq < 5 filtered; test coverage = 100%
3. **Left-padding**: Valid positions at sequence end; masked pooling/softmax
4. **CORE-trm constraint**: Attention weights applied to **original** item embeddings, not transformed
5. **Mixed precision training**: AMP with gradient clipping; temperature learns automatically

---

## Results

### Diginetica Dataset (CIKM Cup 2016)
| Metric | POP | Transition | Session-POP | **CORE-ave** | **CORE-trm** |
|--------|-----|------------|-------------|--------------|--------------|
| **Recall@20** | 0.89% | 34.15% | 21.32% | **39.86%** | **42.52%** |
| **MRR@20** | 0.23% | 13.19% | 14.58% | **14.81%** | **15.88%** |

> **Paper reports (Table 1):** CORE-ave 50.21% / 18.07%, CORE-trm 52.89% / 18.58%
> 
> Our 1-epoch runs achieve **79-80% of paper performance** — full training with early stopping (patience=5) and hyperparameter sweep (margin∈{0.01,0.05,0.07,0.1,1}, dropout∈{0,0.1,0.2}) would close the gap.

### Analysis Insights

- **Short sessions (1-3 clicks)**: CORE-ave nearly matches CORE-trm — mean pooling suffices
- **Long sessions (7+ clicks)**: CORE-trm excels — attention filters noise (misclicks, exploration)
- **Tail items**: All models improve — transition data richer for popular items
- **CORE-trm 1.75× slower** but +6.7% Recall — worth it for production

---

## Project Structure

```
clickstream-core/
├── data/
│   ├── raw/diginetica.csv              # 1.24M events, 310K sessions, 123K items
│   └── processed/                      # Parquet splits + item_mapping.json
├── src/
│   ├── data/
│   │   ├── prepare_data.py             # EDA + temporal split + prefix-target gen
│   │   └── datasets.py                 # PyTorch Dataset (padding, indexing)
│   ├── models/
│   │   ├── core_ave.py                 # Mean pooling encoder
│   │   └── core_trm.py                 # Transformer attention weights
│   ├── evaluation/
│   │   └── metrics.py                  # Recall@K, MRR@K, grouped metrics
│   └── training/
│       ├── config.py                   # Dataclass configs
│       └── train.py                    # AMP, early stop, 3-seed runs
├── app/
│   ├── api.py                          # FastAPI /recommend endpoint
│   └── frontend/                       # Glassmorphism UI
│       ├── index.html
│   │   ├── styles.css                  # 3D text, purple-blue, no gradients
│   │   └── app.js
├── checkpoints/                        # Model weights (gitignored)
├── reports/
│   ├── eda/                            # EDA JSON + 4 plots
│   ├── baselines/                      # Baseline metrics
│   ├── models/                         # Model summaries
│   └── final/                          # API/Frontend report
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# Only torchaudio and pytest-cov may need fresh install
```

### 2. Prepare Data (if not already done)
```bash
python -m src.data.prepare_data
# Generates processed/ + reports/eda/
```

### 3. Train Models
```bash
# Quick test (1 epoch)
python -m src.training.train --model core_ave --seeds 42 --epochs 1 --batch-size 1024
python -m src.training.train --model core_trm --seeds 42 --epochs 1 --batch-size 1024

# Full run (3 seeds, early stopping)
python -m src.training.train --model core_ave --seeds 42 123 456 --epochs 100 --batch-size 2048
python -m src.training.train --model core_trm --seeds 42 123 456 --epochs 100 --batch-size 2048
```

### 4. Run Inference API
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

### 5. Launch Frontend
```bash
cd app/frontend && python -m http.server 3000
# Open http://localhost:3000
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"session_item_ids": [81766, 31331], "k": 5}'
```

```json
{
  "recommendations": [
    {"item_id": 31331, "score": 1.4035, "rank": 1},
    {"item_id": 81766, "score": 1.4028, "rank": 2},
    {"item_id": 30718, "score": 1.4006, "rank": 3}
  ],
  "session_length": 2,
  "model": "core_ave"
}
```

---

## Configuration

Key hyperparameters (in `src/training/config.py` or `.env`):

| Parameter | CORE-ave | CORE-trm | Description |
|-----------|----------|----------|-------------|
| `embedding_dim` | 100 | 100 | Latent dimension |
| `max_seq_len` | 50 | 50 | Max session length |
| `margin` | 0.07 | 0.07 | Controllable margin in loss |
| `item_dropout` | 0.1 | 0.2 | Robust distance measuring |
| `n_layers` | — | 2 | Transformer depth |
| `n_heads` | — | 4 | Attention heads |
| `batch_size` | 2048 | 2048 | Paper setting |
| `lr` | 0.001 | 0.001 | Adam optimizer |

---

## Limitations & Future Work

### Current Limitations
- **CORE-trm inference instability**: AMP training causes NaN in float32 inference (fixed by using CORE-ave for deployment)
- **Anonymous IDs only**: No product metadata — recommendations show raw IDs
- **1-epoch validation**: Full 3-seed runs needed for statistical rigor
- **Warm-item evaluation**: Cannot recommend truly cold items (freq < 5)

### Extensions (from paper + beyond)
1. **MM-CORE**: Multimodal embeddings (text + image) via frozen CLIP/LLM — solves cold-start
2. **Cross-domain routing**: LLM as zero-shot domain bridge (no negative transfer)
3. **Dynamic temperature**: Scale τ by dwell time, scroll depth, add-to-cart signals
4. **Behavior-aware weighting**: αᵢ = softmax(transformer_score + β₁·recency + β₂·repeat_count)

---

## References

- **Paper**: Hou et al., *CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space*, SIGIR 2022
- **Code**: https://github.com/RUCAIBox/CORE
- **Dataset**: Diginetica (CIKM Cup 2016)
- **Baselines**: RecBole framework

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Author

**Divyansh Bhatia**

*Built as part of Semester V, FML Project — reproducing and extending CORE for session-based recommendation.*