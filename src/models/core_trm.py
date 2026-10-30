#!/usr/bin/env python
"""
CORE with Transformer-based attention weighting.

Key differences from CORE-ave:
1. Positional embeddings added to item embeddings
2. Causal Transformer encoder processes sequence
3. Attention weight projection: Linear(dim -> 1) per position
4. Masked softmax over valid positions only
5. Session vector = weighted sum of ORIGINAL item embeddings (not transformed!)
4. Same cosine + temperature + InfoNCE decoding

This is the key innovation: Transformer is used ONLY for weight assignment,
not for transforming the item embeddings themselves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings for session positions."""
    
    def __init__(self, max_len: int, embedding_dim: int):
        super().__init__()
        self.pos_embeddings = nn.Embedding(max_len, embedding_dim)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.01)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return positional embeddings for positions 0 to seq_len-1."""
        positions = torch.arange(seq_len, device=device)
        return self.pos_embeddings(positions)  # [seq_len, dim]


class CausalTransformerEncoder(nn.Module):
    """Causal (masked) Transformer encoder for session encoding."""
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_len: int = 50
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.max_len = max_len
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_len, embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Attention weight projection: maps transformer output to scalar weight per position
        self.weight_projection = nn.Linear(embedding_dim, 1)
        
        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(attention_dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight_projection.weight)
        nn.init.zeros_(self.weight_projection.bias)
    
    def forward(
        self,
        item_embs: torch.Tensor,          # [batch, seq_len, dim]
        prefix_lengths: torch.Tensor      # [batch]
    ) -> torch.Tensor:
        """
        Compute attention weights for each position in the session.
        
        Args:
            item_embs: Item embeddings for the session [batch, seq_len, dim]
            prefix_lengths: Actual lengths of each session [batch]
            
        Returns:
            alpha: Attention weights [batch, seq_len] (sum to 1 over valid positions)
        """
        batch_size, seq_len, dim = item_embs.shape

        pos_emb = self.pos_encoding(seq_len, item_embs.device) 
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  
        
        x = item_embs + pos_emb
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )  
        
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        for i, length in enumerate(prefix_lengths):
            # Valid positions: seq_len - length to seq_len - 1
            valid_start = seq_len - length.item()
            padding_mask[i, valid_start:] = False
        
        # Pass through transformer
        # Combined mask: causal + padding
        # Transformer expects src_key_padding_mask: [batch, seq_len] (True = ignore)
        transformer_out = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # [batch, seq_len, dim]
        
        # Project to scalar weights per position
        weights = self.weight_projection(transformer_out).squeeze(-1)  # [batch, seq_len]
        
        # Apply attention dropout during training
        if self.training:
            weights = self.attn_dropout(weights)
        
        # Masked softmax: only over valid (non-padded) positions
        # Set padding positions to -inf before softmax
        weights = weights.masked_fill(padding_mask, float('-inf'))
        alpha = F.softmax(weights, dim=1)  # [batch, seq_len]
        
        return alpha


class CoreTrm(nn.Module):
    """
    CORE-trm: Representation-Consistent Session Encoder with Transformer Weights.
    
    Session embedding = sum(alpha_i * original_item_embedding_i)
    where alpha_i comes from Transformer attention over the session.
    
    Critical: Uses ORIGINAL item embeddings for weighted sum, not transformed ones.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        padding_idx: int = 0,
        max_len: int = 50,
        item_dropout: float = 0.2,
        init_temperature: float = 1.0,
        margin: float = 0.07,
        min_temperature: float = 0.01,
        # Transformer params
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 200,
        transformer_dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.margin = margin
        self.min_temperature = min_temperature
        
        # Item embedding table (SAME as CORE-ave - shared for encoding & decoding)
        self.item_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        # Item dropout for robust distance measuring
        self.item_dropout = nn.Dropout(item_dropout)
        
        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        
        # Transformer encoder for attention weights
        self.transformer_encoder = CausalTransformerEncoder(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=transformer_dropout,
            attention_dropout=attention_dropout,
            max_len=max_len
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.item_embeddings.weight[self.padding_idx].fill_(0)
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.clamp(self.log_temperature.exp(), min=self.min_temperature)
    
    def encode_session(
        self,
        prefix_indices: torch.Tensor,
        prefix_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode session using Transformer attention weights.
        
        Args:
            prefix_indices: [batch_size, max_len] - item indices (left-padded with 0)
            prefix_lengths: [batch_size] - actual lengths
            
        Returns:
            session_emb: [batch_size, embedding_dim] - L2 normalized
            alpha: [batch_size, max_len] - attention weights (sum to 1 over valid)
        """
        batch_size, max_len = prefix_indices.shape
        
        # Get ORIGINAL item embeddings (for weighted sum later)
        original_item_embs = self.item_embeddings(prefix_indices)  # [batch, max_len, dim]
        
        # Apply item dropout during training
        if self.training:
            original_item_embs = self.item_dropout(original_item_embs)
        
        # Get attention weights from Transformer
        alpha = self.transformer_encoder(original_item_embs, prefix_lengths)  # [batch, max_len]
        
        # Weighted sum of ORIGINAL item embeddings (key CORE constraint!)
        # alpha: [batch, max_len, 1], original_item_embs: [batch, max_len, dim]
        session_emb = (alpha.unsqueeze(-1) * original_item_embs).sum(dim=1)  # [batch, dim]
        
        # L2 normalize for cosine similarity
        session_emb = F.normalize(session_emb, p=2, dim=-1)
        
        return session_emb, alpha
    
    def forward(
        self,
        prefix_indices: torch.Tensor,
        prefix_lengths: torch.Tensor,
        target_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            scores: [batch_size, vocab_size]
            loss: scalar (if target provided)
            alpha: [batch_size, max_len] attention weights
        """
        # Encode session
        session_emb, alpha = self.encode_session(prefix_indices, prefix_lengths)
        
        # Get all item embeddings for scoring
        all_item_embs = self.item_embeddings.weight  # [vocab_size, dim]
        all_item_embs_norm = F.normalize(all_item_embs, p=2, dim=-1)
        
        # Cosine similarity with temperature
        scores = torch.matmul(session_emb, all_item_embs_norm.t()) / self.temperature
        
        # Mask padding
        mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
        scores[:, self.padding_idx] = mask_value
        
        loss = None
        if target_indices is not None:
            loss = self.compute_loss(scores, target_indices)
        
        return scores, loss, alpha
    
    def compute_loss(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss with controllable margin (same as CORE-ave)."""
        batch_size = scores.shape[0]
        pos_scores = scores[torch.arange(batch_size), targets]
        pos_scores_margin = pos_scores + self.margin
        log_sum_exp = torch.logsumexp(scores, dim=1)
        loss = (log_sum_exp - pos_scores_margin).mean()
        return loss
    
    def recommend(
        self,
        prefix_indices: torch.Tensor,
        prefix_lengths: torch.Tensor,
        k: int = 20
    ) -> torch.Tensor:
        """Get top-k recommendations."""
        self.eval()
        with torch.no_grad():
            scores, _, _ = self.forward(prefix_indices, prefix_lengths, target_indices=None)
            top_k = torch.topk(scores, k=k, dim=1)
            return top_k.indices
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get normalized item embeddings."""
        with torch.no_grad():
            embs = self.item_embeddings.weight.clone()
            embs = F.normalize(embs, p=2, dim=-1)
            return embs


def create_core_trm(vocab_size: int, **kwargs) -> CoreTrm:
    """Factory function for CORE-trm."""
    return CoreTrm(vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = create_core_trm(vocab_size=34644).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy batch
    batch_size = 4
    max_len = 50
    prefix = torch.randint(1, 100, (batch_size, max_len)).to(device)
    prefix[:, :20] = 0  # Left padding
    lengths = torch.randint(5, 30, (batch_size,)).to(device)
    targets = torch.randint(1, 1000, (batch_size,)).to(device)
    
    # Forward pass
    scores, loss, alpha = model(prefix, lengths, targets)
    print(f"Scores shape: {scores.shape}")  # [4, 34644]
    print(f"Loss: {loss.item():.4f}")
    print(f"Temperature: {model.temperature.item():.4f}")
    print(f"Alpha shape: {alpha.shape}")  # [4, 50]
    print(f"Alpha sum (per sample): {alpha.sum(dim=1)}")  # Should be ~1.0
    
    # Check padding positions get zero weight
    print(f"Alpha at padding (first 5): {alpha[0, :5]}")
    print(f"Alpha at valid (last 5): {alpha[0, -5:]}")
    
    # Recommend
    topk = model.recommend(prefix, lengths, k=20)
    print(f"Top-20 shape: {topk.shape}")
    
    print("\nOK CORE-trm test passed!")