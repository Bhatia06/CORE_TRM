#!/usr/bin/env python
"""
Milestone D: CORE-ave Model
============================
CORE with mean pooling (uniform weights): session = mean(item_embeddings)

Key components:
1. Item embedding table (shared with CORE-trm)
2. Masked mean pooling over valid positions (ignore padding)
3. L2-normalized session and item vectors
4. Cosine similarity with temperature scaling
5. InfoNCE (cross-entropy) loss with controllable margin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CoreAve(nn.Module):
    """
    CORE-ave: Representation-Consistent Session Encoder with Mean Pooling.
    
    Session embedding = mean of clicked item embeddings (masked for padding)
    Score = cosine_sim(session_emb, item_emb) / temperature
    Loss = InfoNCE with controllable margin
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        padding_idx: int = 0,
        item_dropout: float = 0.1,
        init_temperature: float = 1.0,
        margin: float = 0.07,
        min_temperature: float = 0.01
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.margin = margin
        self.min_temperature = min_temperature
        
        # Item embedding table (shared for encoding and decoding)
        self.item_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        # Item dropout for robust distance measuring (paper)
        self.item_dropout = nn.Dropout(item_dropout)
        
        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with normal distribution."""
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.01)
        # Keep padding at zero
        with torch.no_grad():
            self.item_embeddings.weight[self.padding_idx].fill_(0)
    
    @property
    def temperature(self) -> torch.Tensor:
        """Temperature with minimum bound."""
        return torch.clamp(self.log_temperature.exp(), min=self.min_temperature)
    
    def encode_session(self, prefix_indices: torch.Tensor, prefix_lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode session as masked mean of item embeddings.
        
        Args:
            prefix_indices: [batch_size, max_len] - item indices (0 = padding)
            prefix_lengths: [batch_size] - actual lengths
            
        Returns:
            session_emb: [batch_size, embedding_dim] - L2 normalized
        """
        batch_size, max_len = prefix_indices.shape
        
        # Get item embeddings: [batch_size, max_len, embedding_dim]
        item_embs = self.item_embeddings(prefix_indices)
        
        # Apply item dropout during training (paper: improves robustness)
        if self.training:
            item_embs = self.item_dropout(item_embs)
        
        # Create mask for valid positions: [batch_size, max_len]
        # prefix_indices are left-padded (zeros at start), valid at end
        mask = (prefix_indices != self.padding_idx).float()  # [batch, max_len]
        
        # Masked mean pooling
        # Sum embeddings weighted by mask
        masked_sum = (item_embs * mask.unsqueeze(-1)).sum(dim=1)  # [batch, dim]
        # Divide by actual lengths
        lengths = prefix_lengths.float().unsqueeze(-1).clamp(min=1.0)  # [batch, 1]
        session_emb = masked_sum / lengths  # [batch, dim]
        
        # L2 normalize (for cosine similarity)
        session_emb = F.normalize(session_emb, p=2, dim=-1)
        
        return session_emb
    
    def forward(
        self,
        prefix_indices: torch.Tensor,
        prefix_lengths: torch.Tensor,
        target_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            prefix_indices: [batch_size, max_len]
            prefix_lengths: [batch_size]
            target_indices: [batch_size] - for training loss computation
            
        Returns:
            scores: [batch_size, vocab_size] - cosine similarity / temperature
            loss: scalar (if target_indices provided) or None
        """
        # Encode session
        session_emb = self.encode_session(prefix_indices, prefix_lengths)  # [batch, dim]
        
        # Get all item embeddings (for scoring) - [vocab_size, dim]
        all_item_embs = self.item_embeddings.weight  # Includes padding at 0
        
        # L2 normalize item embeddings
        all_item_embs_norm = F.normalize(all_item_embs, p=2, dim=-1)
        
        # Cosine similarity: [batch, vocab_size]
        scores = torch.matmul(session_emb, all_item_embs_norm.t()) / self.temperature
        
        # Mask padding item (index 0) - set to large negative
        # Use -1e4 instead of -1e9 to avoid half-precision overflow
        mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
        scores[:, self.padding_idx] = mask_value
        
        loss = None
        if target_indices is not None:
            loss = self.compute_loss(scores, target_indices)
        
        return scores, loss
    
    def compute_loss(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE loss with controllable margin.
        
        From paper: L = -log(exp(s_pos + m) / (exp(s_pos + m) + sum(exp(s_neg))))
        where m is the margin.
        
        Args:
            scores: [batch_size, vocab_size] - logits
            targets: [batch_size] - target item indices
            
        Returns:
            loss: scalar
        """
        batch_size = scores.shape[0]
        
        # Get positive scores: [batch_size]
        pos_scores = scores[torch.arange(batch_size), targets]
        
        # Apply margin to positive score (controllable margin from paper)
        pos_scores_margin = pos_scores + self.margin
        
        # Compute log-sum-exp over all items (including positive)
        # This is equivalent to cross-entropy with modified positive logit
        log_sum_exp = torch.logsumexp(scores, dim=1)  # [batch_size]
        
        # Loss = -log(exp(s_pos + m) / sum(exp(s_i)))
        #      = log_sum_exp - (s_pos + m)
        loss = (log_sum_exp - pos_scores_margin).mean()
        
        return loss
    
    def recommend(
        self,
        prefix_indices: torch.Tensor,
        prefix_lengths: torch.Tensor,
        k: int = 20
    ) -> torch.Tensor:
        """
        Get top-k recommendations for a batch of prefixes.
        
        Args:
            prefix_indices: [batch_size, max_len]
            prefix_lengths: [batch_size]
            k: Number of recommendations
            
        Returns:
            top_k_indices: [batch_size, k] - item indices (sorted by score desc)
        """
        self.eval()
        with torch.no_grad():
            scores, _ = self.forward(prefix_indices, prefix_lengths, target_indices=None)
            # Exclude padding (already masked to -1e9)
            top_k = torch.topk(scores, k=k, dim=1)
            return top_k.indices  # [batch_size, k]
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get normalized item embeddings for external use (e.g., vector DB)."""
        with torch.no_grad():
            embs = self.item_embeddings.weight.clone()
            embs = F.normalize(embs, p=2, dim=-1)
            return embs


def create_core_ave(vocab_size: int, **kwargs) -> CoreAve:
    """Factory function for CORE-ave."""
    return CoreAve(vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = create_core_ave(vocab_size=34644).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy batch
    batch_size = 4
    max_len = 50
    prefix = torch.randint(1, 100, (batch_size, max_len)).to(device)
    # Make some positions padding (0)
    prefix[:, :20] = 0
    lengths = torch.randint(5, 30, (batch_size,)).to(device)
    targets = torch.randint(1, 1000, (batch_size,)).to(device)
    
    # Forward pass
    scores, loss = model(prefix, lengths, targets)
    print(f"Scores shape: {scores.shape}")  # [4, 34644]
    print(f"Loss: {loss.item():.4f}")
    print(f"Temperature: {model.temperature.item():.4f}")
    
    # Recommend
    topk = model.recommend(prefix, lengths, k=20)
    print(f"Top-20 shape: {topk.shape}")  # [4, 20]
    
    print("\nOK CORE-ave test passed!")