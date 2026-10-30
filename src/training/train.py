#!/usr/bin/env python
"""
Milestone C/D/E: Training Loop
===============================
- Trains CORE-ave or CORE-trm with early stopping
- Supports 3-seed runs for reliability
- Mixed precision (AMP) for speed
- Gradient clipping
- LR scheduling
- Checkpointing
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from src.data.datasets import create_dataloaders, SessionDataset
from src.models.core_ave import CoreAve, create_core_ave
from src.models.core_trm import CoreTrm, create_core_trm
from src.training.config import (
    ExperimentConfig, get_core_ave_config, get_core_trm_config,
    ModelConfig, TrainingConfig
)
from src.evaluation.metrics import evaluate_model, recall_at_k, mrr_at_k


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Get torch device, fallback to CPU if CUDA not available."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """Create model from config."""
    model_config = config.model
    
    if config.model_type == "core_ave":
        model = create_core_ave(
            vocab_size=model_config.vocab_size,
            embedding_dim=model_config.embedding_dim,
            padding_idx=model_config.padding_idx,
            item_dropout=model_config.item_dropout,
            init_temperature=model_config.init_temperature,
            margin=model_config.margin,
            min_temperature=model_config.min_temperature,
        )
    elif config.model_type == "core_trm":
        model = create_core_trm(
            vocab_size=model_config.vocab_size,
            embedding_dim=model_config.embedding_dim,
            padding_idx=model_config.padding_idx,
            max_len=model_config.max_seq_len,
            item_dropout=model_config.item_dropout,
            init_temperature=model_config.init_temperature,
            margin=model_config.margin,
            min_temperature=model_config.min_temperature,
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
            ff_dim=model_config.ff_dim,
            transformer_dropout=model_config.transformer_dropout,
            attention_dropout=model_config.attention_dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    log_every: int = 100
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    batch_losses = []
    
    progress = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (prefix, target, plen) in enumerate(progress):
        prefix = prefix.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        plen = plen.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                if hasattr(model, 'transformer_encoder'):  # CORE-trm
                    scores, loss, _ = model(prefix, plen, target)
                else:  # CORE-ave
                    scores, loss = model(prefix, plen, target)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if hasattr(model, 'transformer_encoder'):  # CORE-trm
                scores, loss, _ = model(prefix, plen, target)
            else:  # CORE-ave
                scores, loss = model(prefix, plen, target)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        batch_size = prefix.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_losses.append(loss.item())
        
        if (batch_idx + 1) % log_every == 0:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return {"train_loss": avg_loss, "batch_losses": batch_losses}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    k: int = 20,
    use_amp: bool = True
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    all_recall = []
    all_mrr = []
    total_samples = 0
    
    for prefix, target, plen in tqdm(val_loader, desc="Validation", leave=False):
        prefix = prefix.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        plen = plen.to(device, non_blocking=True)
        
        if use_amp:
            with autocast():
                if hasattr(model, 'transformer_encoder'):  # CORE-trm
                    scores, _, _ = model(prefix, plen, target)
                else:  # CORE-ave
                    scores, _ = model(prefix, plen, target)
        else:
            if hasattr(model, 'transformer_encoder'):  # CORE-trm
                scores, _, _ = model(prefix, plen, target)
            else:  # CORE-ave
                scores, _ = model(prefix, plen, target)
        
        # Get top-k predictions
        topk = torch.topk(scores, k=k, dim=1).indices  # [batch, k]
        
        # Compute metrics
        for i in range(prefix.size(0)):
            rec = recall_at_k(topk[i].cpu().tolist(), target[i].item(), k)
            mrr = mrr_at_k(topk[i].cpu().tolist(), target[i].item(), k)
            all_recall.append(rec)
            all_mrr.append(mrr)
        
        total_samples += prefix.size(0)
    
    return {
        "recall@20": float(np.mean(all_recall)),
        "mrr@20": float(np.mean(all_mrr)),
        "n_evaluated": total_samples
    }


def train_model(
    config: ExperimentConfig,
    seed: int,
    checkpoint_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader
) -> Dict[str, Any]:
    """Train a single model with given seed."""
    set_seed(seed)
    device = get_device(config.training.device)
    
    print(f"\n{'='*60}")
    print(f"Training {config.model_type} | Seed: {seed} | Device: {device}")
    print(f"{'='*60}")
    
    # Create model
    model = create_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
        weight_decay=config.training.weight_decay
    )
    
    # LR scheduler
    scheduler = None
    if config.training.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.scheduler_min_lr,
            verbose=True
        )
    
    # AMP scaler
    scaler = GradScaler() if config.training.use_amp and device.type == "cuda" else None
    
    # Training state
    best_metric = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "train_loss": [],
        "val_recall@20": [],
        "val_mrr@20": [],
        "lr": []
    }
    
    # Checkpoint path
    ckpt_path = checkpoint_dir / f"{config.model_type}_seed{seed}_best.pt"
    
    for epoch in range(1, config.training.max_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            use_amp=config.training.use_amp,
            grad_clip=config.training.grad_clip,
            log_every=config.training.log_every_n_batches
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, k=20, use_amp=config.training.use_amp)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_metrics[config.training.monitor_metric])
        
        # Track history
        history["train_loss"].append(train_metrics["train_loss"])
        history["val_recall@20"].append(val_metrics["recall@20"])
        history["val_mrr@20"].append(val_metrics["mrr@20"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Check improvement
        current_metric = val_metrics[config.training.monitor_metric]
        improved = current_metric > best_metric + config.training.min_delta
        
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best checkpoint
            if config.training.save_best_only:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "seed": seed,
                    "best_metric": best_metric,
                    "val_metrics": val_metrics,
                }, ckpt_path)
        else:
            epochs_no_improve += 1
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:3d}/{config.training.max_epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val R@20: {val_metrics['recall@20']:.4f} | "
              f"Val MRR@20: {val_metrics['mrr@20']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {epoch_time:.1f}s"
              f"{'  <- BEST' if improved else ''}")
        
        # Early stopping
        if epochs_no_improve >= config.training.patience:
            print(f"Early stopping at epoch {epoch} (best at epoch {best_epoch})")
            break
    
    # Load best model for final test evaluation
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, k=20, use_amp=config.training.use_amp)
    
    print(f"Test R@20: {test_metrics['recall@20']:.4f}")
    print(f"Test MRR@20: {test_metrics['mrr@20']:.4f}")
    
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "val_metrics": {
            "recall@20": history["val_recall@20"][best_epoch - 1] if best_epoch > 0 else 0,
            "mrr@20": history["val_mrr@20"][best_epoch - 1] if best_epoch > 0 else 0
        },
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint": str(ckpt_path)
    }


def run_experiment(
    model_type: str = "core_ave",
    seeds: Optional[list] = None,
    override_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Run full experiment with multiple seeds.
    
    Args:
        model_type: "core_ave" or "core_trm"
        seeds: List of seeds (default: [42, 123, 456])
        override_config: Dict of config overrides
        
    Returns:
        Dict with results from all seeds + aggregate statistics
    """
    # Get base config
    if model_type == "core_ave":
        config = get_core_ave_config()
    elif model_type == "core_trm":
        config = get_core_trm_config()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Apply overrides
    if override_config:
        for section, values in override_config.items():
            if section == "model" and hasattr(config, "model"):
                for key, value in values.items():
                    if hasattr(config.model, key):
                        setattr(config.model, key, value)
            elif section == "training" and hasattr(config, "training"):
                for key, value in values.items():
                    if hasattr(config.training, key):
                        setattr(config.training, key, value)
            elif hasattr(config, section):
                setattr(config, section, value)
    
    if seeds is not None:
        config.training.seeds = seeds
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir) / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(
        train_path=os.path.join(config.data_dir, "train.parquet"),
        val_path=os.path.join(config.data_dir, "validation.parquet"),
        test_path=os.path.join(config.data_dir, "test.parquet"),
        item_mapping_path=config.item_mapping,
        batch_size=config.training.batch_size,
        max_len=config.model.max_seq_len,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    # Update vocab size in config
    config.model.vocab_size = vocab_size
    
    # Run for each seed
    all_results = []
    
    for seed in config.training.seeds:
        result = train_model(config, seed, checkpoint_dir, train_loader, val_loader, test_loader)
        all_results.append(result)
        
        # Save intermediate results
        results_path = checkpoint_dir / f"results_seed{seed}.json"
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    # Aggregate statistics across seeds
    test_recalls = [r["test_metrics"]["recall@20"] for r in all_results]
    test_mrrs = [r["test_metrics"]["mrr@20"] for r in all_results]
    val_recalls = [r["val_metrics"]["recall@20"] for r in all_results]
    val_mrrs = [r["val_metrics"]["mrr@20"] for r in all_results]
    
    summary = {
        "model_type": model_type,
        "config": config.to_dict(),
        "seeds": config.training.seeds,
        "individual_results": all_results,
        "aggregate": {
            "test_recall@20": {
                "mean": float(np.mean(test_recalls)),
                "std": float(np.std(test_recalls)),
                "values": test_recalls
            },
            "test_mrr@20": {
                "mean": float(np.mean(test_mrrs)),
                "std": float(np.std(test_mrrs)),
                "values": test_mrrs
            },
            "val_recall@20": {
                "mean": float(np.mean(val_recalls)),
                "std": float(np.std(val_recalls)),
                "values": val_recalls
            },
            "val_mrr@20": {
                "mean": float(np.mean(val_mrrs)),
                "std": float(np.std(val_mrrs)),
                "values": val_mrrs
            }
        }
    }
    
    # Save summary
    summary_path = checkpoint_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {model_type}")
    print(f"{'='*60}")
    print(f"Test Recall@20: {summary['aggregate']['test_recall@20']['mean']:.4f} ± {summary['aggregate']['test_recall@20']['std']:.4f}")
    print(f"Test MRR@20:    {summary['aggregate']['test_mrr@20']['mean']:.4f} ± {summary['aggregate']['test_mrr@20']['std']:.4f}")
    print(f"Results saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CORE model")
    parser.add_argument("--model", type=str, default="core_ave", choices=["core_ave", "core_trm"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    
    override = {
        "training": {
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        "model": {
            "margin": args.margin,
            "item_dropout": args.dropout,
        }
    }
    
    run_experiment(model_type=args.model, seeds=args.seeds, override_config=override)