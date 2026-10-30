# Milestone A Report: Data Pipeline & Exploratory Data Analysis

**Date:** 2026-10-28  
**Status:** ✅ COMPLETE  
**Script:** `src/data/prepare_data.py`  
**Runtime:** ~45 seconds

---

## Summary

This milestone implemented the complete data preparation pipeline for the Diginetica session-based recommendation task, following the strict temporal protocol defined in the implementation plan.

### What Was Done

| Step | Description | Output |
|------|-------------|--------|
| 1. **Load & Clean** | Read semicolon-separated CSV, drop duplicates, sort by `sessionId` + `timeframe` | Clean DataFrame (1.24M rows) |
| 2. **EDA** | Comprehensive statistics + 4 required plots | `reports/eda_summary.json`, 4 PNG plots |
| 3. **Temporal Split** | Chronological split by session start date (80/10/10), keeping same-date sessions together | Train/Val/Test splits with zero leakage |
| 4. **Vocabulary** | Built item→index mapping from **train only**, minimum frequency = 5 | `data/processed/item_mapping.json` (34,644 vocab) |
| 5. **Prefix-Target Pairs** | Sliding window per session: `[A,B,C,D]` → `([A],B)`, `([A,B],C)`, `([A,B,C],D)` | 3 Parquet files (725K total pairs) |
| 6. **Reports** | Machine-readable summaries + acceptance checks | `reports/data_split_summary.json` |

---

## Key Findings (What the Data Tells Us)

### 📊 Dataset Characteristics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Sessions** | 310,324 | Large-scale, realistic e-commerce traffic |
| **Avg Session Length** | 3.98 (median: 3) | **Very short sessions** — model must work with 1-3 clicks |
| **Max Session Length** | 87 | Long tail exists but rare (99th percentile = 19) |
| **Unique Items** | 122,993 | Large catalog |
| **Items with <5 interactions** | 78,464 (63.8%) | **Extreme long-tail** — most items are cold |
| **Date Span** | 152 days (Jan–Jun 2016) | Seasonal patterns possible |

### 🔴 Critical Implications for Modeling

1. **Short sessions dominate**: 75% of sessions have ≤5 clicks. The model must make meaningful predictions from 1-3 observed items.
2. **Long-tail is severe**: 64% of items appear <5 times in entire dataset. Even after train-only freq≥5 filtering, vocab is only 34K items (vs 123K raw).
3. **Temporal split is non-negotiable**: Random splitting would leak future trends (e.g., summer items appearing in spring test set). Our chronological split by session start date prevents this.
4. **Zero eval coverage loss**: After filtering rare items, **100% of test targets remain in vocab** — the freq≥5 threshold on train is well-calibrated.

---

## Generated Artifacts

### Processed Data (`data/processed/`)
```
train.parquet          525,484 prefix-target pairs
validation.parquet     102,731 prefix-target pairs  
test.parquet           97,023 prefix-target pairs
item_mapping.json      {item2idx, idx2item, vocab_size=34644}
```

**Parquet Schema:**
```python
session_id: int64
prefix_items: list[int]      # Original item IDs (not indices!)
target_item: int64
prefix_length: int64         # 1, 2, 3, ...
```

### Reports (`reports/`)
```
eda_summary.json                    # All EDA statistics
data_split_summary.json             # Split metadata + acceptance checks
session_length_histogram.png        # Required plot 1
log_item_frequency_histogram.png    # Required plot 2
sessions_per_day_line_chart.png     # Required plot 3
cumulative_share_clicks_by_popularity.png  # Required plot 4
```

---

## Acceptance Checks — All PASSED ✅

| Check | Status | Details |
|-------|--------|---------|
| No session in multiple splits | ✅ | Verified: max unique splits per session = 1 |
| Sequences ordered by timeframe | ✅ | Verified: all sessions monotonic increasing |
| Target appears after prefix | ✅ | By construction (sliding window) |
| Train vocab = candidate vocab | ✅ | Val/test filtered to train vocab only |
| Eval coverage = 100% | ✅ | No test targets removed |

---

## What This Means for Next Steps

### ✅ What You CAN Do Now

| Milestone | What's Ready | What You Need |
|-----------|--------------|---------------|
| **B: Evaluation Utilities** | Prefix-target pairs, vocab mapping | Implement `metrics.py` (Recall@K, MRR@K, grouped metrics) |
| **C: Baselines** | Train item frequencies, session sequences, transitions | Implement POP, Transition, Session-POP in `baselines.py` |
| **D: CORE-ave** | Vocab size (34,644), train pairs, max_seq_len=50 | Implement mean-pooling encoder + cosine + InfoNCE |
| **Quick sanity check** | Load parquet, verify shapes | `pd.read_parquet('data/processed/train.parquet')` |

### ❌ What You CANNOT Do Yet (Blockers)

| Blocked Task | Reason | Unblocked By |
|--------------|--------|--------------|
| Train any neural model | No `datasets.py` (PyTorch Dataset/DataLoader) | Milestone B (or write Dataset class) |
| Run evaluation | No `metrics.py` | Milestone B |
| Compare baselines | No baseline implementations | Milestone C |
| Hyperparameter tuning | No training loop + config | Milestone D + training config |

---

## Recommended Next Step: Milestone B — Evaluation Utilities

Create `src/evaluation/metrics.py` with:

```python
# Required functions:
def recall_at_k(recommended: List[int], target: int, k: int) -> float
def mrr_at_k(recommended: List[int], target: int, k: int) -> float
def evaluate_model(model, test_pairs, k=20) -> Dict[str, float]
def group_by_prefix_length(results, bins=[1, 2-3, 4-6, 7+]) -> Dict
def group_by_target_popularity(results, train_freq, tertiles=True) -> Dict
```

**Unit test requirement**: Hand-verify on tiny synthetic example (per plan §8 Milestone B acceptance check).

---

## Quick Reference: How to Resume

```bash
# Load processed data
import pandas as pd
train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/validation.parquet')
test = pd.read_parquet('data/processed/test.parquet')

# Load vocab mapping
import json
with open('data/processed/item_mapping.json') as f:
    mapping = json.load(f)
item2idx = mapping['item2idx']      # original_id -> 1-based index
idx2item = {int(k): v for k,v in mapping['idx2item'].items()}  # index -> original_id
vocab_size = mapping['vocab_size']  # 34644 (includes padding at 0)

# Convert item IDs to indices for model input
def to_indices(item_list):
    return [item2idx[str(x)] for x in item_list if str(x) in item2idx]
```

---

## Files Modified/Created in This Milestone

```
src/data/prepare_data.py          # Main pipeline script (NEW)
data/processed/
  train.parquet
  validation.parquet
  test.parquet
  item_mapping.json
reports/
  eda_summary.json
  data_split_summary.json
  session_length_histogram.png
  log_item_frequency_histogram.png
  sessions_per_day_line_chart.png
  cumulative_share_clicks_by_popularity.png
reports/milestone_A_report.md     # THIS FILE
```

---

## Notes & Caveats

1. **User IDs are mostly "NA"**: 862,389/1,235,380 (70%) missing — confirms anonymous session setting. Do not use userId.
2. **Timeframe is relative**: Integer timestamps within session, not absolute Unix time. Only `eventdate` gives calendar date.
3. **Prefix items stored as original IDs**: Not indices — convert at DataLoader time (flexible for different vocab experiments).
4. **Max sequence length = 50**: Hardcoded in pipeline; longer prefixes truncated to most recent 50. Matches paper.
5. **No negative sampling in data**: Training uses full softmax over vocab (34K classes) via InfoNCE — handled in model, not data.

---

*Report generated automatically by Milestone A pipeline. For questions, refer to `implementation_plan.md` §4-5, §8 Milestone A.*