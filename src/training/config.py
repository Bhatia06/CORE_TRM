#!/usr/bin/env python
"""
Milestone C/D: Training Configuration
======================================
Centralized hyperparameters for CORE-ave and CORE-trm.
Matches paper settings: dim=100, lr=0.001, batch=2048, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    """Base model configuration."""
    # Vocabulary (set dynamically from data)
    vocab_size: int = 34644  # Includes padding at index 0
    
    # Embedding
    embedding_dim: int = 100
    max_seq_len: int = 50
    padding_idx: int = 0
    
    # Dropout
    embedding_dropout: float = 0.1
    item_dropout: float = 0.2  # For robust distance measuring (paper)
    
    # Temperature (learnable)
    init_temperature: float = 1.0
    min_temperature: float = 0.01
    
    # Margin for controllable loss (paper: grid search {0.01, 0.05, 0.07, 0.1, 1})
    margin: float = 0.07
    
    # CORE-trm specific
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 200  # 2 * embedding_dim
    transformer_dropout: float = 0.1
    attention_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimizer
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Batch & epochs
    batch_size: int = 2048
    max_epochs: int = 100
    
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4
    monitor_metric: str = "mrr@20"  # or "recall@20"
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-5
    
    # Gradient clipping
    grad_clip: float = 1.0
    
    # Mixed precision (for speed)
    use_amp: bool = True
    
    # Seeds for reproducibility (3 seeds per plan)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Device
    device: str = "cuda"  # Will auto-fallback to cpu if not available
    
    # Logging
    log_every_n_batches: int = 100
    save_best_only: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    data_dir: str = "data/processed"
    item_mapping: str = "data/processed/item_mapping.json"
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    # Model type: "core_ave" or "core_trm"
    model_type: str = "core_ave"
    
    # Grid search parameters (for hyperparameter tuning)
    margin_grid: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.07, 0.1, 1.0])
    dropout_grid: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "model_type": self.model_type,
            "margin_grid": self.margin_grid,
            "dropout_grid": self.dropout_grid,
        }


# Default configs for each model
def get_core_ave_config() -> ExperimentConfig:
    """CORE-ave configuration (mean pooling)."""
    return ExperimentConfig(
        model_type="core_ave",
        model=ModelConfig(
            vocab_size=34644,
            embedding_dim=100,
            max_seq_len=50,
            item_dropout=0.1,
            margin=0.07,
        ),
        training=TrainingConfig(
            batch_size=2048,
            learning_rate=0.001,
            patience=5,
            seeds=[42, 123, 456],
        )
    )


def get_core_trm_config() -> ExperimentConfig:
    """CORE-trm configuration (Transformer attention weights)."""
    return ExperimentConfig(
        model_type="core_trm",
        model=ModelConfig(
            vocab_size=34644,
            embedding_dim=100,
            max_seq_len=50,
            item_dropout=0.2,
            margin=0.07,
            n_layers=2,
            n_heads=4,
            ff_dim=200,
            transformer_dropout=0.1,
            attention_dropout=0.1,
        ),
        training=TrainingConfig(
            batch_size=2048,
            learning_rate=0.001,
            patience=5,
            seeds=[42, 123, 456],
        )
    )


if __name__ == "__main__":
    # Print configs
    ave_config = get_core_ave_config()
    trm_config = get_core_trm_config()
    
    import json
    print("=== CORE-ave Config ===")
    print(json.dumps(ave_config.to_dict(), indent=2, default=str))
    print("\n=== CORE-trm Config ===")
    print(json.dumps(trm_config.to_dict(), indent=2, default=str))