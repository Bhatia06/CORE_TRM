#!/usr/bin/env python
"""
Milestone B: Evaluation Utilities
==================================
- Recall@K, MRR@K computations
- Grouped metrics by prefix length and target item popularity
- Synthetic unit test for hand-verification
- Unified evaluation loop for any recommender model
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import defaultdict
import numpy as np
import pandas as pd


# ─── Core Metrics ────────────────────────────────────────────────

def recall_at_k(recommended: List[int], target: int, k: int) -> float:
    """
    Recall@K: 1 if target in top-k recommendations, else 0.
    
    Args:
        recommended: Ranked list of candidate item IDs (length >= k)
        target: Ground truth next item ID
        k: Cutoff (e.g., 20)
    
    Returns:
        1.0 if target in recommended[:k], else 0.0
    """
    if not recommended or k <= 0:
        return 0.0
    return 1.0 if target in recommended[:k] else 0.0


def mrr_at_k(recommended: List[int], target: int, k: int) -> float:
    """
    MRR@K: Mean Reciprocal Rank at K.
    
    Args:
        recommended: Ranked list of candidate item IDs (length >= k)
        target: Ground truth next item ID
        k: Cutoff (e.g., 20)
    
    Returns:
        1/rank if target in top-k, else 0.0
        (rank is 1-indexed: rank=1 -> 1.0, rank=5 -> 0.2)
    """
    if not recommended or k <= 0:
        return 0.0
    try:
        rank = recommended[:k].index(target) + 1  # 1-indexed
        return 1.0 / rank
    except ValueError:
        return 0.0


def hit_at_k(recommended: List[int], target: int, k: int) -> float:
    """Alias for recall_at_k (same computation)."""
    return recall_at_k(recommended, target, k)


# ─── Grouped Metrics ─────────────────────────────────────────────

def group_by_prefix_length(
    results: List[Dict],
    bins: Optional[List[Tuple[int, int]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Group metrics by prefix length bins.
    
    Args:
        results: List of dicts with keys: 'prefix_length', 'recall@20', 'mrr@20'
        bins: List of (min_len, max_len) tuples. Default: [(1,1), (2,3), (4,6), (7, inf)]
    
    Returns:
        Dict mapping bin_label -> {'recall@20': mean, 'mrr@20': mean, 'count': n}
    """
    if bins is None:
        bins = [(1, 1), (2, 3), (4, 6), (7, float('inf'))]
    
    bin_labels = []
    for lo, hi in bins:
        if hi == float('inf'):
            bin_labels.append(f"{lo}+")
        elif lo == hi:
            bin_labels.append(str(lo))
        else:
            bin_labels.append(f"{lo}-{hi}")
    
    grouped = {label: {'recall@20': [], 'mrr@20': [], 'count': 0} for label in bin_labels}
    
    for r in results:
        plen = r['prefix_length']
        for (lo, hi), label in zip(bins, bin_labels):
            if lo <= plen <= hi:
                grouped[label]['recall@20'].append(r.get('recall@20', 0.0))
                grouped[label]['mrr@20'].append(r.get('mrr@20', 0.0))
                grouped[label]['count'] += 1
                break
    
    # Compute means
    summary = {}
    for label, vals in grouped.items():
        n = vals['count']
        if n > 0:
            summary[label] = {
                'recall@20': float(np.mean(vals['recall@20'])),
                'mrr@20': float(np.mean(vals['mrr@20'])),
                'count': n
            }
        else:
            summary[label] = {'recall@20': 0.0, 'mrr@20': 0.0, 'count': 0}
    
    return summary


def group_by_target_popularity(
    results: List[Dict],
    train_item_freq: Dict[int, int],
    n_tertiles: int = 3,
    labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Group metrics by target item popularity (train frequency tertiles).
    
    Args:
        results: List of dicts with keys: 'target_item', 'recall@20', 'mrr@20'
        train_item_freq: Dict mapping item_id -> frequency in train set
        n_tertiles: Number of buckets (default 3: head, torso, tail)
        labels: Custom labels for buckets (default: ['head', 'torso', 'tail'])
    
    Returns:
        Dict mapping bucket -> {'recall@20': mean, 'mrr@20': mean, 'count': n, 'freq_range': (min, max)}
    """
    if labels is None:
        labels = ['head', 'torso', 'tail'][:n_tertiles]
    
    # Get frequency for each target in results
    target_freqs = []
    for r in results:
        tid = r['target_item']
        freq = train_item_freq.get(tid, 0)
        target_freqs.append(freq)
    
    if not target_freqs:
        return {label: {'recall@20': 0.0, 'mrr@20': 0.0, 'count': 0, 'freq_range': (0, 0)} for label in labels}
    
    # Compute tertile boundaries
    sorted_freqs = sorted(set(target_freqs))
    n_unique = len(sorted_freqs)
    
    if n_unique <= n_tertiles:
        # Few unique frequencies - assign each to own bucket
        boundaries = sorted_freqs
    else:
        boundaries = np.percentile(sorted_freqs, [100 * i / n_tertiles for i in range(1, n_tertiles)])
    
    # Assign each result to bucket
    buckets = {label: {'recall@20': [], 'mrr@20': [], 'count': 0, 'freqs': []} for label in labels}
    
    for r, freq in zip(results, target_freqs):
        # Find bucket
        bucket_idx = 0
        for i, b in enumerate(boundaries):
            if freq <= b:
                bucket_idx = i
                break
        else:
            bucket_idx = n_tertiles - 1
        
        label = labels[bucket_idx]
        buckets[label]['recall@20'].append(r.get('recall@20', 0.0))
        buckets[label]['mrr@20'].append(r.get('mrr@20', 0.0))
        buckets[label]['count'] += 1
        buckets[label]['freqs'].append(freq)
    
    # Compute summary
    summary = {}
    for label, vals in buckets.items():
        n = vals['count']
        if n > 0:
            summary[label] = {
                'recall@20': float(np.mean(vals['recall@20'])),
                'mrr@20': float(np.mean(vals['mrr@20'])),
                'count': n,
                'freq_range': (int(min(vals['freqs'])), int(max(vals['freqs'])))
            }
        else:
            summary[label] = {
                'recall@20': 0.0, 'mrr@20': 0.0, 'count': 0, 'freq_range': (0, 0)
            }
    
    return summary


# ─── Model Evaluation Interface ──────────────────────────────────

class RecommenderModel:
    """
    Abstract base for recommender models.
    Implement recommend(prefix_items: List[int], k: int) -> List[int]
    """
    def recommend(self, prefix_items: List[int], k: int) -> List[int]:
        raise NotImplementedError


def evaluate_model(
    model: RecommenderModel,
    test_pairs: pd.DataFrame,
    k: int = 20,
    item2idx: Optional[Dict[str, int]] = None,
    idx2item: Optional[Dict[int, int]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a recommender model on test prefix-target pairs.
    
    Args:
        model: Object with recommend(prefix_items, k) -> List[int] method
        test_pairs: DataFrame with columns ['session_id', 'prefix_items', 'target_item', 'prefix_length']
        k: Cutoff for metrics
        item2idx: Optional mapping for models that use indices internally
        idx2item: Optional reverse mapping
        verbose: Print progress
    
    Returns:
        Dict with overall metrics + per-example results + grouped metrics
    """
    results = []
    total = len(test_pairs)
    
    for i, row in test_pairs.iterrows():
        prefix = row['prefix_items']
        target = row['target_item']
        plen = row['prefix_length']
        
        # Get recommendations
        if item2idx is not None and idx2item is not None:
            # Model expects indices, convert prefix
            prefix_idx = [item2idx[str(x)] for x in prefix if str(x) in item2idx]
            rec_idx = model.recommend(prefix_idx, k)
            # Convert back to item IDs
            recommended = [idx2item.get(idx, -1) for idx in rec_idx]
        else:
            # Model works with original item IDs directly
            recommended = model.recommend(prefix, k)
        
        # Compute metrics
        r_at_k = recall_at_k(recommended, target, k)
        mrr_k = mrr_at_k(recommended, target, k)
        
        results.append({
            'session_id': row['session_id'],
            'prefix_length': plen,
            'target_item': target,
            'recommended': recommended,
            'recall@20': r_at_k,
            'mrr@20': mrr_k
        })
        
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Evaluated {i+1}/{total} ({(i+1)/total*100:.1f}%)")
    
    # Overall metrics
    overall_recall = float(np.mean([r['recall@20'] for r in results]))
    overall_mrr = float(np.mean([r['mrr@20'] for r in results]))
    
    # Grouped metrics
    by_prefix_len = group_by_prefix_length(results)
    
    return {
        'overall': {
            'recall@20': overall_recall,
            'mrr@20': overall_mrr,
            'n_evaluated': total
        },
        'by_prefix_length': by_prefix_len,
        'per_example': results
    }


# ─── Synthetic Unit Test (Hand-Verification) ─────────────────────

def run_synthetic_test():
    """
    Hand-verification on tiny synthetic example.
    Per plan §8 Milestone B acceptance check.
    """
    print("=" * 60)
    print("SYNTHETIC UNIT TEST - Hand Verification")
    print("=" * 60)
    
    # Synthetic test case
    # Prefix: [A, B], Target: C
    # Model recommends: [C, D, E, F, ...] -> rank 1
    # Model recommends: [D, C, E, F, ...] -> rank 2
    # Model recommends: [D, E, F, ...] -> not in top-k
    
    test_cases = [
        # (recommended, target, k, expected_recall, expected_mrr)
        ([10, 20, 30, 40], 10, 20, 1.0, 1.0),      # Target at rank 1
        ([10, 20, 30, 40], 20, 20, 1.0, 0.5),      # Target at rank 2
        ([10, 20, 30, 40], 30, 20, 1.0, 1/3),      # Target at rank 3
        ([10, 20, 30, 40], 50, 20, 0.0, 0.0),      # Target not in top-k
        ([10, 20, 30, 40], 10, 1, 1.0, 1.0),       # k=1, target at rank 1
        ([10, 20, 30, 40], 20, 1, 0.0, 0.0),       # k=1, target at rank 2 (excluded)
        ([], 10, 20, 0.0, 0.0),                    # Empty recommendations
    ]
    
    all_passed = True
    for i, (rec, target, k, exp_recall, exp_mrr) in enumerate(test_cases):
        r = recall_at_k(rec, target, k)
        m = mrr_at_k(rec, target, k)
        
        recall_ok = abs(r - exp_recall) < 1e-9
        mrr_ok = abs(m - exp_mrr) < 1e-9
        passed = recall_ok and mrr_ok
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  Test {i+1}: {status}")
        print(f"    recommended={rec}, target={target}, k={k}")
        print(f"    recall@k: {r:.4f} (expected {exp_recall:.4f})")
        print(f"    mrr@k:    {m:.4f} (expected {exp_mrr:.4f})")
    
    # Test grouped metrics
    print("\n  Testing group_by_prefix_length...")
    synthetic_results = [
        {'prefix_length': 1, 'recall@20': 1.0, 'mrr@20': 1.0},
        {'prefix_length': 1, 'recall@20': 0.0, 'mrr@20': 0.0},
        {'prefix_length': 2, 'recall@20': 1.0, 'mrr@20': 0.5},
        {'prefix_length': 3, 'recall@20': 1.0, 'mrr@20': 0.33},
        {'prefix_length': 5, 'recall@20': 0.0, 'mrr@20': 0.0},
        {'prefix_length': 10, 'recall@20': 1.0, 'mrr@20': 0.25},
    ]
    grouped = group_by_prefix_length(synthetic_results)
    print(f"    Groups: {list(grouped.keys())}")
    for g, v in grouped.items():
        print(f"      {g}: recall={v['recall@20']:.2f}, mrr={v['mrr@20']:.2f}, n={v['count']}")
    
    # Test popularity grouping
    print("\n  Testing group_by_target_popularity...")
    train_freq = {10: 1000, 20: 500, 30: 100, 40: 50, 50: 10, 60: 5}
    pop_results = [
        {'target_item': 10, 'recall@20': 1.0, 'mrr@20': 1.0},  # head
        {'target_item': 20, 'recall@20': 1.0, 'mrr@20': 0.5},  # head
        {'target_item': 30, 'recall@20': 0.5, 'mrr@20': 0.25}, # torso
        {'target_item': 40, 'recall@20': 0.0, 'mrr@20': 0.0},  # torso
        {'target_item': 50, 'recall@20': 0.0, 'mrr@20': 0.0},  # tail
        {'target_item': 60, 'recall@20': 1.0, 'mrr@20': 1.0},  # tail
    ]
    pop_grouped = group_by_target_popularity(pop_results, train_freq)
    for g, v in pop_grouped.items():
        print(f"      {g}: recall={v['recall@20']:.2f}, mrr={v['mrr@20']:.2f}, n={v['count']}, freq_range={v['freq_range']}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")
    
    return all_passed


# ─── Baseline Model Wrappers (for evaluation) ────────────────────

class PopularityRecommender(RecommenderModel):
    """POP: Recommend globally most popular items from train."""
    def __init__(self, train_item_freq: Dict[int, int], k: int = 20):
        self.top_items = [item for item, _ in 
                          sorted(train_item_freq.items(), key=lambda x: -x[1])][:k]
    
    def recommend(self, prefix_items: List[int], k: int) -> List[int]:
        return self.top_items[:k]


class TransitionRecommender(RecommenderModel):
    """Item-to-item transition: recommend items most often after last item."""
    def __init__(self, transitions: Dict[int, Dict[int, int]], k: int = 20):
        # transitions[last_item][next_item] = count
        self.transitions = transitions
        self.k = k
        # Precompute sorted next items for each last_item
        self.sorted_next = {}
        for last, next_counts in transitions.items():
            self.sorted_next[last] = [item for item, _ in 
                                       sorted(next_counts.items(), key=lambda x: -x[1])]
    
    def recommend(self, prefix_items, k: int) -> List[int]:
        # Handle both list and numpy array
        if hasattr(prefix_items, '__len__') and len(prefix_items) == 0:
            return []
        last = prefix_items[-1]
        return self.sorted_next.get(last, [])[:k]


class SessionPopularityRecommender(RecommenderModel):
    """Session-POP: rank by frequency in current session, backfill with POP."""
    def __init__(self, train_item_freq: Dict[int, int], k: int = 20):
        self.train_freq = train_item_freq
        self.pop_items = [item for item, _ in 
                          sorted(train_item_freq.items(), key=lambda x: -x[1])][:k]
        self.k = k
    
    def recommend(self, prefix_items, k: int) -> List[int]:
        # Handle both list and numpy array
        if hasattr(prefix_items, '__len__') and len(prefix_items) == 0:
            return []
        # Count frequency in current session
        from collections import Counter
        sess_counts = Counter(prefix_items)
        # Sort by session frequency (desc), then by global popularity
        sess_sorted = sorted(sess_counts.keys(), key=lambda x: (-sess_counts[x], 
                            -self.train_freq.get(x, 0)))
        # Backfill with global POP
        recommended = []
        for item in sess_sorted:
            if item not in recommended:
                recommended.append(item)
        for item in self.pop_items:
            if item not in recommended:
                recommended.append(item)
        return recommended[:k]


# ─── Main for Running Tests ──────────────────────────────────────

if __name__ == "__main__":
    run_synthetic_test()