# Milestone F Report: FastAPI Backend & Glassmorphism Frontend

**Date:** 2026-10-30  
**Status:** ✅ COMPLETE  
**Components:**
- `app/api.py` — FastAPI REST API with `/recommend` endpoint
- `app/frontend/index.html` — Semantic HTML5 structure
- `app/frontend/styles.css` — Glassmorphism UI, purple-blue theme, 3D text
- `app/frontend/app.js` — Vanilla JS, session management, API integration

---

## Summary

Built a complete inference serving stack for the trained CORE model:
- **Backend**: FastAPI with automatic model loading, input validation, error handling
- **Frontend**: Single-page glassmorphism UI with catalog, session panel, recommendations
- **Theme**: Purple-blue, no gradients, 3D text shadows, subtle animations

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health, model info, vocab size, device |
| `/recommend` | POST | Get top-k recommendations for a session |
| `/` | GET | Service info & example request |

### Request Format
```json
POST /recommend
{
  "session_item_ids": [81766, 31331, 32118],
  "k": 20
}
```

### Response Format
```json
{
  "recommendations": [
    {"item_id": 32118, "score": 1.4029, "rank": 1},
    {"item_id": 81766, "score": 1.4025, "rank": 2},
    ...
  ],
  "session_length": 3,
  "model": "core_ave"
}
```

---

## Frontend Features

### Catalog Panel (Left)
- 47 items from Diginetica vocabulary (freq ≥ 5)
- Clickable cards with hover/tap feedback
- Keyboard accessible (Enter/Space to add)
- Staggered entrance animation

### Session Panel (Center)
- Chronological list of clicked items
- Numbered badges, item names + IDs
- Live length counter with 3D text effect
- Clear session button (trash icon)

### Recommendations Panel (Right)
- Ranked list with scores (4 decimal places)
- Rank badges in primary color
- Score pills with success color
- Loading spinner during inference
- Empty state with helpful messaging

---

## Design System

### Colors (CSS Variables)
```css
--bg-deep: #0a0614           /* Deep navy base */
--primary: #8b7df6           /* Electric purple */
--accent: #c4b5fd            /* Light lavender */
--text-primary: #f0edfc      /* Near white */
--text-secondary: #b8b0d0    /* Muted lavender */
--text-muted: #7a7094        /* Subtle purple-gray */
```

### Glassmorphism
```css
background: rgba(18, 12, 32, 0.65);
border: 1px solid rgba(120, 100, 180, 0.18);
backdrop-filter: blur(20px) saturate(180%);
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
```

### 3D Text Effect
```css
text-shadow: 
  0 1px 0 #7a6ce8,
  0 2px 0 #6b5fd8,
  0 3px 0 #5d53c8,
  0 4px 0 #4f47b8,
  0 5px 0 #423da8,
  0 6px 12px rgba(139, 125, 246, 0.4),
  0 12px 24px rgba(139, 125, 246, 0.2);
```

---

## Technical Details

### Model Loading (Startup)
```python
MODEL = create_core_ave(vocab_size=34644)
checkpoint = torch.load("checkpoints/core_ave/core_ave_seed42_best.pt")
MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.to(DEVICE).eval()
```

### Inference Pipeline
1. Convert item IDs → indices (filter unknown)
2. Left-pad to max_len=50
3. Forward pass → cosine similarity scores
4. Top-k with `.float()` for numerical stability
5. Convert indices → original item IDs
6. Return ranked list with scores

### Error Handling
- 400: No valid items in session (all unknown)
- 500: Model inference errors (with message)
- CORS enabled for all origins

---

## Responsive Layout

| Breakpoint | Grid |
|------------|------|
| < 768px | Single column (stacked) |
| 768-1199px | 3-column: 1.2fr / 1fr / 1.3fr |
| ≥ 1200px | Fixed: 380px / 300px / 1fr |

---

## Usage

```bash
# Start API server
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000

# Open frontend
# Serve app/frontend/ with any static server:
#   npx serve app/frontend
#   python -m http.server -d app/frontend 3000
# Or open app/frontend/index.html directly in browser
```

---

## Verification

| Test | Result |
|------|--------|
| Health check | ✅ 200 OK |
| Valid session (3 items) | ✅ 200, 10 recs with scores |
| Session items in vocab | ✅ Top ranks include session items |
| Unknown item | ✅ 400 with clear message |
| Empty session | Frontend shows empty state |
| CORS | ✅ Enabled for browser access |
| Keyboard nav | ✅ Enter/Space on catalog items |

---

## Notes

- **CORE-ave used**: CORE-trm has float32 inference NaN issues from AMP training
- **Scores are cosine similarity / temperature** — higher = more similar
- **Session items reappear in recommendations** — model learns repetition patterns
- **Frontend is static** — no build step, works from file:// or any static host
- **Font**: Space Grotesk (UI) + JetBrains Mono (IDs, scores) via Google Fonts

---

*API + Frontend complete. Full project pipeline operational.*