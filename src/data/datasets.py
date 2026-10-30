#!/usr/bin/env python
"""
Milestone C/D: PyTorch Dataset & DataLoader for Session-Based Recommendation
============================================================================
- Converts prefix item IDs to indices (using item_mapping.json)
- Pads/truncates sequences to max_len
- Returns tensors ready for model input
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class SessionDataset(Dataset):
    """
    Dataset for session-based next-item prediction.
    
    Each sample: (prefix_indices, target_index, prefix_length)
    - prefix_indices: LongTensor [max_len] padded with 0 (padding idx)
    - target_index: LongTensor [] (single item index)
    - prefix_length: LongTensor [] (actual length before padding)
    """
    
    def __init__(
        self,
        parquet_path: str,
        item_mapping_path: str,
        max_len: int = 50,
        min_prefix_len: int = 1
    ):
        """
        Args:
            parquet_path: Path to train/val/test.parquet
            item_mapping_path: Path to item_mapping.json
            max_len: Maximum sequence length (pad/truncate to this)
            min_prefix_len: Minimum prefix length to keep (filter shorter)
        """
        self.max_len = max_len
        self.min_prefix_len = min_prefix_len
        
        # Load data
        self.df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(self.df)} samples from {parquet_path}")
        
        # Load item mapping
        with open(item_mapping_path, 'r') as f:
            mapping = json.load(f)
        self.item2idx = mapping['item2idx']  # str(item_id) -> idx (1-based)
        self.idx2item = {int(k): v for k, v in mapping['idx2item'].items()}
        self.vocab_size = mapping['vocab_size']  # includes padding at 0
        
        # Filter by prefix length
        original_len = len(self.df)
        self.df = self.df[self.df['prefix_length'] >= min_prefix_len].reset_index(drop=True)
        if len(self.df) < original_len:
            print(f"Filtered {original_len - len(self.df)} samples with prefix_length < {min_prefix_len}")
        
        print(f"Vocab size: {self.vocab_size}, Max len: {self.max_len}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        prefix_items = row['prefix_items']  # List of original item IDs
        target_item = row['target_item']
        prefix_length = row['prefix_length']
        
        # Convert prefix to indices (1-based, 0 = padding)
        prefix_indices = []
        for item_id in prefix_items:
            idx = self.item2idx.get(str(item_id))
            if idx is not None:
                prefix_indices.append(idx)
        
        # Truncate to max_len (keep most recent)
        if len(prefix_indices) > self.max_len:
            prefix_indices = prefix_indices[-self.max_len:]
            prefix_length = self.max_len
        
        # Pad to max_len
        padded = [0] * (self.max_len - len(prefix_indices)) + prefix_indices
        
        # Target index
        target_idx = self.item2idx.get(str(target_item), 0)
        
        return (
            torch.tensor(padded, dtype=torch.long),      # [max_len]
            torch.tensor(target_idx, dtype=torch.long),  # []
            torch.tensor(prefix_length, dtype=torch.long) # []
        )
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_num_items(self) -> int:
        return self.vocab_size - 1  # Exclude padding


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    item_mapping_path: str,
    batch_size: int = 2048,
    max_len: int = 50,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train/val/test DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, vocab_size
    """
    train_dataset = SessionDataset(train_path, item_mapping_path, max_len=max_len)
    val_dataset = SessionDataset(val_path, item_mapping_path, max_len=max_len)
    test_dataset = SessionDataset(test_path, item_mapping_path, max_len=max_len)
    
    vocab_size = train_dataset.get_vocab_size()
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    print(f"Vocab size (incl padding): {vocab_size}")
    
    return train_loader, val_loader, test_loader, vocab_size


# Quick test
if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(
        'data/processed/train.parquet',
        'data/processed/validation.parquet',
        'data/processed/test.parquet',
        'data/processed/item_mapping.json',
        batch_size=32,  # Small for test
        max_len=50
    )
    
    # Inspect one batch
    for batch in train_loader:
        prefix, target, plen = batch
        print(f"\nBatch shapes:")
        print(f"  prefix: {prefix.shape} (should be [32, 50])")
        print(f"  target: {target.shape} (should be [32])")
        print(f"  plen:   {plen.shape} (should be [32])")
        print(f"\nSample prefix (first 10): {prefix[0][:10]}")
        print(f"Sample target: {target[0].item()}")
        print(f"Sample plen: {plen[0].item()}")
        break
    
    print("\nOK Dataset test passed!")