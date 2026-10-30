#!/usr/bin/env python
"""Full baseline evaluation with results saved to JSON."""

import json
import pandas as pd
from collections import Counter

from src.evaluation.metrics import (
    PopularityRecommender, TransitionRecommender, SessionPopularityRecommender,
    evaluate_model, group_by_target_popularity
)

# Load data
train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/validation.parquet')
test = pd.read_parquet('data/processed/test.parquet')

with open('data/processed/item_mapping.json') as f:
    m = json.load(f)
item2idx = m['item2idx']
idx2item = {int(k): v for k, v in m['idx2item'].items()}

# Build train item frequency
train_items = []
for row in train.itertuples():
    train_items.extend(row.prefix_items)
    train_items.append(row.target_item)
train_freq = Counter(train_items)

# Build transitions
transitions = {}
for row in train.itertuples():
    if len(row.prefix_items) > 0:
        last = row.prefix_items[-1]
        nxt = row.target_item
        if last not in transitions:
            transitions[last] = Counter()
        transitions[last][nxt] += 1

# Evaluate all baselines
models = {
    'POP': PopularityRecommender(train_freq, k=20),
    'Transition': TransitionRecommender(transitions, k=20),
    'Session-POP': SessionPopularityRecommender(train_freq, k=20),
}

results = {}
for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    res = evaluate_model(model, test, k=20, verbose=True)
    # Add popularity grouping
    res['by_target_popularity'] = group_by_target_popularity(res['per_example'], train_freq)
    results[name] = res
    print(f"{name}: Recall@20={res['overall']['recall@20']:.4f}, MRR@20={res['overall']['mrr@20']:.4f}")

# Save results
output = {}
for name, res in results.items():
    output[name] = {
        'overall': res['overall'],
        'by_prefix_length': res['by_prefix_length'],
        'by_target_popularity': res['by_target_popularity']
    }

with open('reports/baseline_metrics.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n\nSaved to reports/baseline_metrics.json")
print("\n=== SUMMARY ===")
for name, res in results.items():
    print(f"{name:15s} | Recall@20: {res['overall']['recall@20']:.4f} | MRR@20: {res['overall']['mrr@20']:.4f}")