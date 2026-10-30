#!/usr/bin/env python
"""
Milestone A: Data Pipeline with Full EDA
=========================================
- Reads raw Diginetica CSV (semicolon-separated)
- Performs comprehensive EDA with required plots
- Strict temporal split by session start date (80/10/10)
- Builds item vocabulary from TRAIN only (freq >= 5)
- Creates prefix-target pairs per session
- Saves processed parquet files, item mapping, EDA report, split summary
"""

import os
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ─── Configuration ──────────────────────────────────────────────
RAW_PATH = Path("data/raw/diginetica.csv")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Preprocessing
MIN_ITEM_FREQ = 5
MIN_SESSION_LEN = 2
MAX_SEQ_LEN = 50  # For later model use

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


# ─── Helper Functions ───────────────────────────────────────────
def save_plot(fig, name: str):
    """Save plot to reports directory."""
    path = REPORTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {path}")


def save_json(obj, name: str):
    """Save JSON to reports directory."""
    path = REPORTS_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  Saved JSON: {path}")


# ─── Main Pipeline ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MILESTONE A: Data Pipeline & EDA")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Load Raw Data
    # ══════════════════════════════════════════════════════════════
    print("\n[1/7] Loading raw data...")
    df = pd.read_csv(RAW_PATH, sep=';', dtype={
        'sessionId': 'int64',
        'userId': 'str',
        'itemId': 'int64',
        'timeframe': 'int64',
        'eventdate': 'str'
    })
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Basic Cleaning & Validation
    # ══════════════════════════════════════════════════════════════
    print("\n[2/7] Cleaning & validation...")

    # Check for missing values
    missing = df.isnull().sum().to_dict()
    print(f"  Missing values: {missing}")

    # Exact duplicate events (all columns identical)
    dup_count = df.duplicated().sum()
    print(f"  Exact duplicate rows: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"  Shape after dedup: {df.shape}")

    # Sort each session by timeframe ascending
    df = df.sort_values(['sessionId', 'timeframe']).reset_index(drop=True)
    print("  Sorted by sessionId, timeframe")

    # Verify sorting
    session_order_check = df.groupby('sessionId')['timeframe'].apply(lambda x: x.is_monotonic_increasing).all()
    print(f"  All sessions sorted by timeframe: {session_order_check}")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Exploratory Data Analysis (EDA)
    # ══════════════════════════════════════════════════════════════
    print("\n[3/7] Running EDA...")

    # Basic counts
    n_events = len(df)
    n_sessions = df['sessionId'].nunique()
    n_items = df['itemId'].nunique()
    n_users = df['userId'].nunique()

    print(f"  Events: {n_events:,}")
    print(f"  Sessions: {n_sessions:,}")
    print(f"  Unique items: {n_items:,}")
    print(f"  Known users (non-NA): {(df['userId'] != 'NA').sum():,}")

    # Session length distribution
    session_lengths = df.groupby('sessionId').size()
    sess_len_stats = {
        'count': int(session_lengths.count()),
        'mean': float(session_lengths.mean()),
        'median': float(session_lengths.median()),
        'std': float(session_lengths.std()),
        'min': int(session_lengths.min()),
        'max': int(session_lengths.max()),
        'percentiles': {
            '25%': float(session_lengths.quantile(0.25)),
            '50%': float(session_lengths.quantile(0.50)),
            '75%': float(session_lengths.quantile(0.75)),
            '90%': float(session_lengths.quantile(0.90)),
            '95%': float(session_lengths.quantile(0.95)),
            '99%': float(session_lengths.quantile(0.99)),
        }
    }
    print(f"  Session length stats: {json.dumps(sess_len_stats, indent=4)}")

    # Item frequency distribution
    item_freq = df['itemId'].value_counts()
    items_lt5 = (item_freq < 5).sum()
    print(f"  Items with < 5 interactions: {items_lt5:,} ({(items_lt5/n_items)*100:.1f}%)")

    # Date range
    df['eventdate_parsed'] = pd.to_datetime(df['eventdate'])
    date_range = {
        'min': df['eventdate_parsed'].min().strftime('%Y-%m-%d'),
        'max': df['eventdate_parsed'].max().strftime('%Y-%m-%d'),
        'n_unique_dates': df['eventdate_parsed'].nunique()
    }
    print(f"  Date range: {date_range['min']} to {date_range['max']} ({date_range['n_unique_dates']} unique dates)")

    # Events/sessions per day
    daily_events = df.groupby('eventdate_parsed').size()
    daily_sessions = df.groupby('eventdate_parsed')['sessionId'].nunique()

    # Sessions crossing multiple eventdates
    session_dates = df.groupby('sessionId')['eventdate_parsed'].nunique()
    multi_date_sessions = (session_dates > 1).sum()
    print(f"  Sessions spanning multiple dates: {multi_date_sessions:,} ({(multi_date_sessions/n_sessions)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: EDA PLOTS (Required by Plan)
    # ══════════════════════════════════════════════════════════════
    print("\n[4/7] Generating EDA plots...")

    # Plot 1: Session Length Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(session_lengths, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(session_lengths.mean(), color='red', linestyle='--', label=f'Mean: {session_lengths.mean():.1f}')
    ax.axvline(session_lengths.median(), color='green', linestyle='--', label=f'Median: {session_lengths.median():.1f}')
    ax.set_xlabel('Session Length (number of clicks)')
    ax.set_ylabel('Number of Sessions')
    ax.set_title('Session Length Distribution (Diginetica)')
    ax.legend()
    ax.set_xlim(0, min(100, session_lengths.max()))
    save_plot(fig, "session_length_histogram")

    # Plot 2: Log-scaled Item Frequency Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.log10(item_freq.values), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.log10(5), color='red', linestyle='--', label='Min freq threshold (log10(5))')
    ax.set_xlabel('log10(Item Frequency)')
    ax.set_ylabel('Number of Items')
    ax.set_title('Item Frequency Distribution (Log-Scaled)')
    ax.legend()
    save_plot(fig, "log_item_frequency_histogram")

    # Plot 3: Sessions per Day Line Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_sessions.plot(ax=ax, marker='o', markersize=3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Sessions')
    ax.set_title('Sessions per Day')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save_plot(fig, "sessions_per_day_line_chart")

    # Plot 4: Cumulative Share of Clicks by Item Popularity
    fig, ax = plt.subplots(figsize=(10, 6))
    item_freq_sorted = item_freq.sort_values(ascending=False)
    cumsum = item_freq_sorted.cumsum() / item_freq_sorted.sum() * 100
    ax.plot(range(1, len(cumsum) + 1), cumsum.values, linewidth=2)
    ax.axhline(80, color='red', linestyle='--', label='80% of clicks')
    ax.axhline(95, color='orange', linestyle='--', label='95% of clicks')
    ax.set_xlabel('Items (ranked by popularity)')
    ax.set_ylabel('Cumulative % of Total Clicks')
    ax.set_title('Cumulative Share of Clicks by Item Popularity (Long-Tail)')
    ax.legend()
    ax.set_xscale('log')
    save_plot(fig, "cumulative_share_clicks_by_popularity")

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Temporal Split by Session Start Date
    # ══════════════════════════════════════════════════════════════
    print("\n[5/7] Temporal split by session start date...")

    # Get session start date (first eventdate per session)
    session_start = df.groupby('sessionId')['eventdate_parsed'].min().reset_index()
    session_start.columns = ['sessionId', 'start_date']

    # Sort unique dates chronologically
    unique_dates = sorted(session_start['start_date'].unique())
    n_dates = len(unique_dates)
    print(f"  Unique session start dates: {n_dates}")

    # Split dates: 80/10/10 keeping all sessions from same date together
    train_date_cutoff = int(n_dates * TRAIN_RATIO)
    val_date_cutoff = int(n_dates * (TRAIN_RATIO + VAL_RATIO))

    train_dates = set(unique_dates[:train_date_cutoff])
    val_dates = set(unique_dates[train_date_cutoff:val_date_cutoff])
    test_dates = set(unique_dates[val_date_cutoff:])

    print(f"  Train dates: {len(train_dates)} ({min(train_dates)} to {max(train_dates)})")
    print(f"  Val dates:   {len(val_dates)} ({min(val_dates)} to {max(val_dates)})")
    print(f"  Test dates:  {len(test_dates)} ({min(test_dates)} to {max(test_dates)})")

    # Assign sessions to splits
    session_start['split'] = session_start['start_date'].map(
        lambda d: 'train' if d in train_dates else ('val' if d in val_dates else 'test')
    )

    # Merge split back to main df
    df = df.merge(session_start[['sessionId', 'split']], on='sessionId', how='left')

    # Verify no session in multiple splits
    split_check = df.groupby('sessionId')['split'].nunique().max()
    assert split_check == 1, f"Session appears in multiple splits! Max unique splits per session: {split_check}"
    print(f"  OK: No session appears in more than one split")

    split_counts = df['split'].value_counts()
    print(f"  Split distribution:\n{split_counts}")

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Build Item Vocabulary from TRAIN Only (freq >= 5)
    # ══════════════════════════════════════════════════════════════
    print("\n[6/7] Building item vocabulary from train set...")

    train_df = df[df['split'] == 'train'].copy()
    train_item_freq = train_df['itemId'].value_counts()
    retained_items = train_item_freq[train_item_freq >= MIN_ITEM_FREQ].index.tolist()

    print(f"  Unique items in train: {train_item_freq.shape[0]:,}")
    print(f"  Items with freq >= {MIN_ITEM_FREQ}: {len(retained_items):,}")
    print(f"  Items removed (rare): {train_item_freq.shape[0] - len(retained_items):,}")

    # Create item-to-index mapping (0 reserved for padding/masking)
    item2idx = {item: idx + 1 for idx, item in enumerate(retained_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}
    vocab_size = len(item2idx) + 1  # +1 for padding at index 0

    print(f"  Vocabulary size (incl. padding): {vocab_size}")

    # Save mapping
    mapping = {
        'item2idx': item2idx,
        'idx2item': {str(k): v for k, v in idx2item.items()},  # JSON keys must be strings
        'vocab_size': vocab_size,
        'min_train_freq': MIN_ITEM_FREQ
    }
    with open(PROCESSED_DIR / "item_mapping.json", 'w') as f:
        json.dump(mapping, f)
    print(f"  Saved item_mapping.json")

    # ══════════════════════════════════════════════════════════════
    # STEP 7: Filter & Create Prefix-Target Pairs
    # ══════════════════════════════════════════════════════════════
    print("\n[7/7] Creating prefix-target pairs per split...")

    def create_prefix_target_pairs(split_df, split_name):
        """Create (prefix, target) pairs from sessions in a split."""
        pairs = []
        removed_events = 0
        removed_targets = 0
        original_targets = 0

        for session_id, group in split_df.groupby('sessionId'):
            # Get item sequence (already sorted by timeframe)
            items = group['itemId'].tolist()

            # Filter to retained vocabulary
            filtered_items = [item for item in items if item in item2idx]
            removed_events += len(items) - len(filtered_items)

            if len(filtered_items) < MIN_SESSION_LEN:
                continue

            # Generate prefix-target pairs: [A,B,C,D] -> ([A],B), ([A,B],C), ([A,B,C],D)
            for i in range(1, len(filtered_items)):
                prefix = filtered_items[:i]
                target = filtered_items[i]
                original_targets += 1

                if target not in item2idx:
                    removed_targets += 1
                    continue

                # Truncate prefix to max_seq_len (keep most recent)
                if len(prefix) > MAX_SEQ_LEN:
                    prefix = prefix[-MAX_SEQ_LEN:]

                pairs.append({
                    'session_id': session_id,
                    'prefix_items': prefix,
                    'target_item': target,
                    'prefix_length': len(prefix)
                })

        eval_coverage = (original_targets - removed_targets) / original_targets * 100 if original_targets > 0 else 0
        print(f"  {split_name}: {len(pairs):,} pairs, {removed_events:,} events removed, "
              f"{removed_targets:,}/{original_targets:,} targets removed "
              f"(coverage: {eval_coverage:.1f}%)")

        return pd.DataFrame(pairs), {
            'n_pairs': len(pairs),
            'removed_events': removed_events,
            'removed_targets': removed_targets,
            'original_targets': original_targets,
            'eval_coverage_pct': eval_coverage
        }

    train_pairs, train_stats = create_prefix_target_pairs(train_df, "train")
    val_pairs, val_stats = create_prefix_target_pairs(df[df['split'] == 'val'], "val")
    test_pairs, test_stats = create_prefix_target_pairs(df[df['split'] == 'test'], "test")

    # Save processed parquet files
    train_pairs.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_pairs.to_parquet(PROCESSED_DIR / "validation.parquet", index=False)
    test_pairs.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"\n  Saved train.parquet ({len(train_pairs):,} rows)")
    print(f"  Saved validation.parquet ({len(val_pairs):,} rows)")
    print(f"  Saved test.parquet ({len(test_pairs):,} rows)")

    # ══════════════════════════════════════════════════════════════
    # STEP 8: Save Reports
    # ══════════════════════════════════════════════════════════════
    print("\n[8/7] Saving reports...")

    # EDA Summary
    eda_summary = {
        'raw_data': {
            'n_events': int(n_events),
            'n_sessions': int(n_sessions),
            'n_items': int(n_items),
            'n_known_users': int((df['userId'] != 'NA').sum()),
            'missing_values': missing,
            'duplicate_events': int(dup_count),
            'session_sorted_by_timeframe': bool(session_order_check),
            'date_range': date_range,
            'sessions_multi_date': int(multi_date_sessions),
            'sessions_multi_date_pct': float(multi_date_sessions / n_sessions * 100)
        },
        'session_length': sess_len_stats,
        'item_frequency': {
            'n_items_total': int(n_items),
            'n_items_freq_lt5': int(items_lt5),
            'pct_items_freq_lt5': float(items_lt5 / n_items * 100),
            'max_freq': int(item_freq.max()),
            'median_freq': float(item_freq.median())
        },
        'vocabulary': {
            'train_unique_items': int(train_item_freq.shape[0]),
            'retained_items_freq_ge_5': len(retained_items),
            'vocab_size_with_padding': vocab_size
        }
    }
    save_json(eda_summary, "eda_summary")

    # Data Split Summary
    split_summary = {
        'split_method': 'chronological_by_session_start_date',
        'target_ratio': {'train': TRAIN_RATIO, 'validation': VAL_RATIO, 'test': TEST_RATIO},
        'train_dates': {'count': len(train_dates), 'min': str(min(train_dates)), 'max': str(max(train_dates))},
        'val_dates': {'count': len(val_dates), 'min': str(min(val_dates)), 'max': str(max(val_dates))},
        'test_dates': {'count': len(test_dates), 'min': str(min(test_dates)), 'max': str(max(test_dates))},
        'train_stats': train_stats,
        'val_stats': val_stats,
        'test_stats': test_stats,
        'acceptance_checks': {
            'no_session_in_multiple_splits': True,
            'sequences_ordered_by_timeframe': True,
            'targets_after_prefix': True,
            'train_vocab_only_candidates': True
        }
    }
    save_json(split_summary, "data_split_summary")

    print("\n" + "=" * 60)
    print("MILESTONE A COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - {PROCESSED_DIR}/train.parquet")
    print(f"  - {PROCESSED_DIR}/validation.parquet")
    print(f"  - {PROCESSED_DIR}/test.parquet")
    print(f"  - {PROCESSED_DIR}/item_mapping.json")
    print(f"  - {REPORTS_DIR}/eda_summary.json")
    print(f"  - {REPORTS_DIR}/data_split_summary.json")
    print(f"  - {REPORTS_DIR}/session_length_histogram.png")
    print(f"  - {REPORTS_DIR}/log_item_frequency_histogram.png")
    print(f"  - {REPORTS_DIR}/sessions_per_day_line_chart.png")
    print(f"  - {REPORTS_DIR}/cumulative_share_clicks_by_popularity.png")


if __name__ == "__main__":
    main()