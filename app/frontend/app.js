// CORE Session Recommender - Frontend Application
// Glassmorphism UI with 3D text effects, purple-blue theme

const API_BASE = 'http://localhost:8000';
const MAX_SESSION_LENGTH = 50;

// ─── Sample Catalog Data ──────────────────────────────────────
// Using actual item IDs from Diginetica that are in the vocabulary (freq >= 5)
const CATALOG_ITEMS = [
  { id: 81766, name: "Wireless Headphones Pro" },
  { id: 31331, name: "Bluetooth Speaker Mini" },
  { id: 32118, name: "USB-C Hub 7-in-1" },
  { id: 9654, name: "Mechanical Keyboard RGB" },
  { id: 5103, name: "Ergonomic Mouse Vertical" },
  { id: 69597, name: "27\" 4K Monitor IPS" },
  { id: 4168, name: "Laptop Stand Aluminum" },
  { id: 79898, name: "Webcam 1080p Streaming" },
  { id: 36083, name: "External SSD 1TB NVMe" },
  { id: 10858, name: "WiFi 6 Router AX3000" },
  { id: 14614, name: "Smart Desk Lamp LED" },
  { id: 6848, name: "Cable Management Kit" },
  { id: 79141, name: "Monitor Arm Dual" },
  { id: 12345, name: "Gaming Chair Ergonomic" },
  { id: 34567, name: "Keycap Set PBT Cherry" },
  { id: 78901, name: "RGB Light Strip 2m" },
  { id: 89012, name: "Microphone Arm Boom" },
  { id: 1234, name: "Acoustic Panels 12pk" },
  { id: 5678, name: "Stream Deck Mini" },
  { id: 9012, name: "Green Screen Collapsible" },
  { id: 23456, name: "Mechanical Switch Tester" },
  { id: 45678, name: "Wrist Rest Memory Foam" },
  { id: 56789, name: "Desk Mat XL Extended" },
  { id: 67890, name: "Headphone Stand Wood" },
  { id: 90123, name: "Pop Filter Studio" },
  { id: 3456, name: "Ring Light 18\"" },
  { id: 7890, name: "Capture Card 4K60" },
  { id: 1245, name: "Audio Interface 2i2" },
  { id: 6789, name: "XLR Cable 3m" },
  { id: 1122, name: "Studio Monitors 5\"" },
  { id: 3344, name: "MIDI Controller 25" },
  { id: 5566, name: "Synthesizer Module" },
  { id: 7788, name: "Audio Cable TRS" },
  { id: 9900, name: "Patch Bay 48pt" },
  { id: 2233, name: "Rack Mount 12U" },
  { id: 4455, name: "Power Conditioner" },
  { id: 6677, name: "UPS 1500VA" },
  { id: 8899, name: "Network Switch 8pt" },
  { id: 1010, name: "Patch Cable Cat6" },
  { id: 2020, name: "Fiber Optic Cable" },
  { id: 3030, name: "Server Rail Kit" },
  { id: 4040, name: "KVM Switch 4pt" },
  { id: 5050, name: "Console Cable RJ45" },
  { id: 6060, name: "Label Maker Pro" },
  { id: 7070, name: "Cable Tester RJ45" },
  { id: 8080, name: "Crimping Tool Set" },
  { id: 9090, name: "Fish Tape 50ft" },
];

// ─── State ─────────────────────────────────────────────────────
let sessionItems = [];
let isLoading = false;

// ─── DOM Elements ──────────────────────────────────────────────
const catalogGrid = document.getElementById('catalogGrid');
const sessionSequence = document.getElementById('sessionSequence');
const sessionLength = document.getElementById('sessionLength');
const recList = document.getElementById('recList');
const recLoading = document.getElementById('recLoading');
const resetBtn = document.getElementById('resetBtn');
const modelBadge = document.getElementById('modelBadge');

// ─── Utility Functions ─────────────────────────────────────────
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showLoading(show) {
  isLoading = show;
  recLoading.hidden = !show;
  if (show) {
    recList.innerHTML = '';
  }
}

// ─── Render Functions ──────────────────────────────────────────
function renderCatalog() {
  catalogGrid.innerHTML = CATALOG_ITEMS.map((item, index) => `
    <button class="catalog-item" 
            data-id="${item.id}" 
            data-name="${escapeHtml(item.name)}"
            aria-label="Add ${escapeHtml(item.name)} to session"
            style="animation-delay: ${index * 15}ms">
      <div class="catalog-item-name">${escapeHtml(item.name)}</div>
      <div class="catalog-item-id">ID: ${item.id}</div>
    </button>
  `).join('');

  // Add click handlers
  catalogGrid.querySelectorAll('.catalog-item').forEach(btn => {
    btn.addEventListener('click', () => addToSession(
      parseInt(btn.dataset.id, 10),
      btn.dataset.name
    ));
  });
}

function renderSession() {
  if (sessionItems.length === 0) {
    sessionSequence.innerHTML = '<p class="session-empty">Click items from catalog to start a session</p>';
  } else {
    sessionSequence.innerHTML = `
      <div class="session-items">
        ${sessionItems.map((item, index) => `
          <div class="session-item" style="animation-delay: ${index * 50}ms">
            <span class="session-item-number">${index + 1}</span>
            <div class="session-item-info">
              <div class="session-item-name">${escapeHtml(item.name)}</div>
              <div class="session-item-id">ID: ${item.id}</div>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }
  sessionLength.textContent = sessionItems.length;
}

function renderRecommendations(recommendations) {
  if (!recommendations || recommendations.length === 0) {
    recList.innerHTML = `
      <div class="rec-empty">
        <svg class="rec-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/>
          <path d="M21 21l-4.35-4.35"/>
        </svg>
        <p>No recommendations available</p>
      </div>
    `;
    return;
  }

  recList.innerHTML = recommendations.map((rec, index) => `
    <div class="rec-item" style="animation-delay: ${index * 40}ms">
      <span class="rec-rank">${rec.rank}</span>
      <div class="rec-item-info">
        <div class="rec-item-name">${escapeHtml(rec.name || `Item ${rec.item_id}`)}</div>
        <div class="rec-item-meta">
          <span class="rec-item-id">ID: ${rec.item_id}</span>
          <span class="rec-score">${rec.score.toFixed(4)}</span>
        </div>
      </div>
    </div>
  `).join('');
}

// ─── Session Management ────────────────────────────────────────
function addToSession(itemId, itemName) {
  if (sessionItems.length >= MAX_SESSION_LENGTH) {
    sessionItems.shift(); // Remove oldest
  }
  sessionItems.push({ id: itemId, name: itemName });
  renderSession();
  fetchRecommendations();
}

function clearSession() {
  sessionItems = [];
  renderSession();
  recList.innerHTML = `
    <div class="rec-empty">
      <svg class="rec-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="11" cy="11" r="8"/>
        <path d="M21 21l-4.35-4.35"/>
      </svg>
      <p>Start a session to see recommendations</p>
    </div>
  `;
}

// ─── API Communication ────────────────────────────────────────
async function fetchRecommendations() {
  if (sessionItems.length === 0) return;
  
  showLoading(true);
  
  try {
    const response = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_item_ids: sessionItems.map(i => i.id),
        k: 20
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Enrich with names from catalog
    const enriched = data.recommendations.map(rec => {
      const catalogItem = CATALOG_ITEMS.find(c => c.id === rec.item_id);
      return { ...rec, name: catalogItem?.name };
    });
    
    renderRecommendations(enriched);
    modelBadge.textContent = data.model.toUpperCase();
    
  } catch (error) {
    console.error('Recommendation error:', error);
    recList.innerHTML = `
      <div class="rec-empty">
        <svg class="rec-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        <p>Failed to load recommendations</p>
        <p style="font-size: 0.75rem; color: var(--text-muted);">${error.message}</p>
      </div>
    `;
  } finally {
    showLoading(false);
  }
}

// ─── Health Check ──────────────────────────────────────────────
async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (response.ok) {
      const data = await response.json();
      modelBadge.textContent = data.model.toUpperCase();
      console.log('API connected:', data);
    }
  } catch (error) {
    console.warn('API not available:', error.message);
    modelBadge.textContent = 'OFFLINE';
  }
}

// ─── Initialization ────────────────────────────────────────────
function init() {
  renderCatalog();
  renderSession();
  checkHealth();
  
  resetBtn.addEventListener('click', clearSession);
  
  // Keyboard support for catalog
  catalogGrid.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      const btn = e.target.closest('.catalog-item');
      if (btn) {
        e.preventDefault();
        btn.click();
      }
    }
  });
  
  console.log('CORE Recommender UI initialized');
}

// Start when DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}