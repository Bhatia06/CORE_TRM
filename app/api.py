#!/usr/bin/env python
"""
FastAPI Backend for Session-Based Recommendation
=================================================
POST /recommend endpoint that uses trained CORE model.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import model classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models.core_ave import CoreAve, create_core_ave
from src.models.core_trm import CoreTrm, create_core_trm


# ─── Configuration ──────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
MODEL_TYPE = "core_ave"  # core_ave works reliably; core_trm has float32 inference issues
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load item mapping
ITEM_MAPPING_PATH = Path("data/processed/item_mapping.json")
with open(ITEM_MAPPING_PATH, 'r') as f:
    ITEM_MAPPING = json.load(f)

ITEM2IDX = ITEM_MAPPING['item2idx']  # str(item_id) -> idx
IDX2ITEM = {int(k): v for k, v in ITEM_MAPPING['idx2item'].items()}
VOCAB_SIZE = ITEM_MAPPING['vocab_size']
PADDING_IDX = 0
MAX_LEN = 50


# ─── Model Loading ──────────────────────────────────────────────
def load_model():
    """Load the best trained model checkpoint."""
    if MODEL_TYPE == "core_ave":
        model = create_core_ave(vocab_size=VOCAB_SIZE)
        ckpt_path = CHECKPOINT_DIR / "core_ave" / f"core_ave_seed{SEED}_best.pt"
    else:
        model = create_core_trm(vocab_size=VOCAB_SIZE)
        ckpt_path = CHECKPOINT_DIR / "core_trm" / f"core_trm_seed{SEED}_best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {MODEL_TYPE} from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


# Load model at startup
MODEL = load_model()


# ─── Request/Response Models ────────────────────────────────────
class RecommendRequest(BaseModel):
    session_item_ids: List[int] = Field(..., description="List of item IDs in current session (chronological)")
    k: int = Field(20, ge=1, le=50, description="Number of recommendations to return")


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    rank: int


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]
    session_length: int
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    vocab_size: int
    device: str


# ─── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="CORE Session-Based Recommender",
    description="Representation-Consistent Session Recommendation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper Functions ───────────────────────────────────────────
def encode_session(session_items: List[int]) -> tuple:
    """Convert session item IDs to padded tensor."""
    # Convert to indices (filter unknown items)
    indices = []
    for item_id in session_items:
        idx = ITEM2IDX.get(str(item_id))
        if idx is not None:
            indices.append(idx)
    
    if not indices:
        raise HTTPException(status_code=400, detail="No valid items in session (all items unknown)")
    
    # Truncate to max_len (keep most recent)
    if len(indices) > MAX_LEN:
        indices = indices[-MAX_LEN:]
    
    # Left-pad with zeros
    padded = [0] * (MAX_LEN - len(indices)) + indices
    prefix_tensor = torch.tensor([padded], dtype=torch.long).to(DEVICE)
    length_tensor = torch.tensor([len(indices)], dtype=torch.long).to(DEVICE)
    
    return prefix_tensor, length_tensor


# ─── API Endpoints ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model=MODEL_TYPE,
        vocab_size=VOCAB_SIZE,
        device=str(DEVICE)
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get top-k recommendations for a session."""
    try:
        # Encode session
        prefix_tensor, length_tensor = encode_session(request.session_item_ids)
        
        # Get recommendations
        with torch.no_grad():
            if hasattr(MODEL, 'transformer_encoder'):
                scores, _, _ = MODEL(prefix_tensor, length_tensor)
            else:
                scores, _ = MODEL(prefix_tensor, length_tensor)
            
            # Get top-k (excluding padding at index 0)
            topk_scores, topk_indices = torch.topk(scores.float(), k=request.k, dim=1)
            topk_scores = topk_scores[0].cpu().tolist()
            topk_indices = topk_indices[0].cpu().tolist()
        
        # Convert indices back to item IDs
        recommendations = []
        for rank, (idx, score) in enumerate(zip(topk_indices, topk_scores), 1):
            if idx == PADDING_IDX:
                continue
            item_id = IDX2ITEM.get(idx)
            if item_id is not None:
                recommendations.append(RecommendationItem(
                    item_id=item_id,
                    score=round(score, 4),
                    rank=rank
                ))
        
        return RecommendResponse(
            recommendations=recommendations,
            session_length=len(request.session_item_ids),
            model=MODEL_TYPE
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "service": "CORE Session-Based Recommender",
        "model": MODEL_TYPE,
        "endpoints": {
            "health": "GET /health",
            "recommend": "POST /recommend"
        },
        "example_request": {
            "session_item_ids": [123, 456, 789],
            "k": 20
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)