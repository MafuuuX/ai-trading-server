# AI Trading Server

Distributed model training server for 62 stocks with a **universal BiGRU model**, 18 ticker-agnostic technical features, self-attention, and dual-head (classification + regression) architecture.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Universal Model](#universal-model)
- [Technical Indicators (18 Features)](#technical-indicators-18-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Stock Coverage](#stock-coverage)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Features

✅ **Universal BiGRU Model** – One model trained across all 62 tickers simultaneously  
✅ **18 Ticker-Agnostic Features** – All indicators normalised as percentages/ratios (no absolute prices)  
✅ **Dual-Head Architecture** – Simultaneous 3-class classification (UP/NEUTRAL/DOWN) + regression  
✅ **Self-Attention + Residual** – Attention-weighted context vector with skip connection  
✅ **Class-Weight Balancing** – Automatic sample weighting to combat label imbalance  
✅ **Multi-Provider Price Fetching** – Alpaca (primary) → Finnhub → Yahoo (fallback)  
✅ **Parallel Price Collection** – ThreadPoolExecutor with up to 20 workers  
✅ **WebSocket Support** – Real-time training progress & server events  
✅ **Modern Dashboard** – Dark-themed web UI with live training progress  
✅ **Google Drive Backup** – Automatic daily model backups (keeps last 7)  
✅ **Smart Caching** – 2-hour TTL with automatic validation  
✅ **Zero-Downtime Training** – Background jobs don't block API  
✅ **Legacy Per-Ticker Models** – Fallback LSTM models still supported  

---

## Architecture

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Data Fetcher** | `data_fetcher.py` | Multi-provider price data (Alpaca → Finnhub → Yahoo) with rate limiting & caching |
| **Universal Trainer** | `universal_trainer.py` | BiGRU(128)→BiGRU(64) + attention + residual, 18 features, 60-day lookback |
| **Legacy Trainer** | `trainer.py` | Per-ticker dual-head LSTM (fallback) |
| **API Server** | `server.py` | FastAPI with REST + WebSocket, JWT auth, model serving, dashboard |
| **Dashboard** | `templates/dashboard.html` | Modern dark-themed UI with live training status |
| **Risk Profiles** | `risk_profiles.py` | Per-user risk configuration |
| **Simulator** | `simulator.py` | Backtesting engine |

### Data Flow

```
Market Data (Alpaca/Finnhub/Yahoo)
        │
        ▼
  CachedDataFetcher (2h TTL)
        │
        ▼
  UniversalModelTrainer
    ├── Feature Engineering (18 indicators)
    ├── Z-score normalisation + clipping (±5σ)
    ├── BiGRU(128) → BiGRU(64) + Attention
    └── Dual Head: Classification (3-class) + Regression
        │
        ▼
  Model Files (H5 + Scaler PKL)
        │
        ▼
  FastAPI Server → Client Download / WebSocket Updates
```

---

## Universal Model

The universal model replaces per-ticker training with a single model trained across all 62 stocks. This gives more training data, better generalisation, and eliminates the need to maintain hundreds of individual models.

### Architecture (v2)

```
Input (60, 18)
    │
    ▼
BiGRU(128, return_sequences=True) + LayerNorm
    │
    ▼
BiGRU(64, return_sequences=True) + LayerNorm
    │
    ├──► Self-Attention → Context Vector (weighted sum)
    ├──► Last Hidden State
    │
    ▼
Concatenate([Context, LastHidden])  ← residual connection
    │
    ▼
Dense(128, ReLU) + LayerNorm + Dropout(0.2)
    │
    ▼
Dense(64, ReLU) + Dropout(0.15)
    │
    ├──► Classification Head (Dense 3, Softmax) → UP / NEUTRAL / DOWN
    └──► Regression Head (Dense 1, Linear) → predicted % change
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Lookback window | 60 days |
| Epochs | 80 (EarlyStopping patience=15) |
| Batch size | 64 |
| Learning rate | 5×10⁻⁴ (ReduceLROnPlateau, factor=0.5, patience=7) |
| Classification loss | SparseCategoricalCrossentropy |
| Regression loss | Huber |
| Loss weights | Classification: 1.0, Regression: 0.3 |
| Label threshold | ±0.4% (UP/DOWN vs NEUTRAL boundary) |
| Z-score clipping | ±5.0 |
| Train/Val split | 85% / 15% |
| Data shuffling | Cross-ticker random permutation |
| Class balancing | Per-sample weights (auto-computed) |

---

## Technical Indicators (18 Features)

All features are **ticker-agnostic** – expressed as percentages, ratios, or oscillator values. No absolute prices.

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 1 | `Close_pct` | Close price % change | ~ ±5% |
| 2 | `High_pct` | High price % change | ~ ±5% |
| 3 | `Low_pct` | Low price % change | ~ ±5% |
| 4 | `Volume_pct` | Volume % change | ~ ±100% |
| 5 | `RSI` | Relative Strength Index (14-period) | 0–100 |
| 6 | `MACD_norm` | MACD normalised by price (%) | ~ ±2% |
| 7 | `BB_position` | Price position within Bollinger Bands | 0–100% |
| 8 | `BB_width` | Bollinger Band width as % of price | ~ 2–15% |
| 9 | `ATR_pct` | Average True Range as % of price | ~ 1–5% |
| 10 | `SMA20_dist` | Distance from SMA(20) as % | ~ ±10% |
| 11 | `SMA50_dist` | Distance from SMA(50) as % | ~ ±15% |
| 12 | `Stoch_K` | Stochastic %K (14,3) | 0–100 |
| 13 | `Stoch_D` | Stochastic %D (signal) | 0–100 |
| 14 | `ROC_10` | Rate of Change (10-period) | ~ ±20% |
| 15 | `Williams_R` | Williams %R (14-period) | -100–0 |
| 16 | `CCI` | Commodity Channel Index (20-period) | ~ ±300 |
| 17 | `OBV_pct` | On-Balance Volume % change (clipped ±50) | ±50% |
| 18 | `Momentum` | 10-day price momentum (%) | ~ ±20% |

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/MafuuuX/ai-trading-server.git
cd ai-trading-server
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow` ≥ 2.13 – Deep learning
- `ta` – Technical analysis indicators
- `pandas` / `numpy` – Data processing
- `scikit-learn` – ML utilities
- `yfinance` – Market data (fallback)
- `fastapi` / `uvicorn` – API server
- `jinja2` – Dashboard templates

### 4. (Optional) Systemd Service
```bash
sudo cp ai-trading-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-server
sudo systemctl start ai-trading-server
```

---

## Configuration

### Environment Variables
```bash
export ADMIN_USER="admin"
export ADMIN_PASS="your_secure_password"
export SESSION_SECRET="your_session_secret_key"

# API Keys (for multi-provider price fetching)
export ALPACA_API_KEY_ID="your_alpaca_key"
export ALPACA_SECRET_KEY="your_alpaca_secret"
export ALPACA_BASE_URL="https://data.alpaca.markets"
export FINNHUB_API_KEY="your_finnhub_key"
```

Or create `api_keys.py` in the parent directory:
```python
def load_api_keys():
    return {
        'alpaca': 'your_key',
        'alpaca_secret': 'your_secret',
        'alpaca_base_url': 'https://data.alpaca.markets',
        'finnhub': 'your_key',
    }
```

---

## Usage

### Start Server
```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Access Dashboard
Open `http://localhost:8000` – login with ADMIN_USER / ADMIN_PASS.

The dashboard shows:
- **Universal Model** status card with live training progress bar
- System stats (CPU, RAM, model count, queue)
- Per-ticker model grid (collapsible)

### Train Universal Model
```bash
# Via API
curl -X POST http://localhost:8000/api/train-all

# Via dashboard – click "Train Universal Model"
```

### Train Legacy Per-Ticker Models
```bash
curl -X POST http://localhost:8000/api/train-all-legacy
```

---

## API Documentation

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server health & uptime |

### Universal Model

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/train-all` | Start universal model training |
| `GET` | `/api/training-status/universal` | Training progress (status, epoch, accuracy) |
| `GET` | `/api/models/universal/download` | Download model + scaler as ZIP |
| `GET` | `/api/models/universal/hash` | SHA256 hash for update detection |

### Legacy Per-Ticker Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/train-all-legacy` | Queue all per-ticker training |
| `POST` | `/api/train/{ticker}` | Train a single ticker |
| `POST` | `/api/train-batch` | Train a batch of tickers |
| `GET` | `/api/training-status` | Per-ticker training status |
| `GET` | `/api/models/{ticker}/download` | Download ticker model ZIP |
| `GET` | `/api/models/{ticker}/hash` | Ticker model hash |

### Live Prices

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/prices/{ticker}` | Single stock live price |
| `GET` | `/api/prices?tickers=AAPL,MSFT` | Multiple stock prices |
| `GET` | `/api/chart-cache` | All cached live prices |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `WS /ws` | Real-time training progress, server status, price updates |

### Trade Journal

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/trades` | Get trade journal |
| `POST` | `/api/trades` | Log a trade |
| `POST` | `/api/trades/close` | Close a trade |
| `GET` | `/api/trades/analytics` | Trading analytics & win rates |

---

## Stock Coverage

**62 stocks** across 8 sectors + ETFs:

| Sector | Tickers |
|--------|---------|
| **Mega Cap Tech** | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, GOOG, NFLX, AMD |
| **Finance** | JPM, BAC, WFC, GS, MS, BLK, SCHW, COIN, V, MA, AXP |
| **Healthcare** | UNH, JNJ, LLY, MRK, PFE, ABBV, TMO, AMGN |
| **Energy** | CVX, XOM, COP, MPC, PSX, VLO, HES, SLB |
| **Consumer** | MCD, SBUX, NKE, LULU, WMT, KO, PEP, HD |
| **ETFs** | SPY, QQQ, IWM, EEM, XLK, XLV, XLF, XLE |
| **Semiconductors & Enterprise** | AVGO, ORCL, INTC, CRM, ADBE, IBM, CSCO, QCOM, TXN |

---

## Performance

| Metric | Value |
|--------|-------|
| **Hardware** | i5-6400 (4C/4T), 24 GB RAM |
| **Universal training time** | ~30–60 min (62 tickers, 80 epochs) |
| **Model file size** | ~5 MB (H5) + ~200 KB (scaler) |
| **API response time** | < 100 ms |
| **Price collection** | ~5s cycle (parallel, 20 workers) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Too Many Requests"** | Multi-provider fallback handles this automatically. Check API keys are configured. |
| **WebSocket 403** | Ensure `allow_credentials=False` in CORS config and `websockets` package installed. |
| **No price data** | Check API keys in env vars or `api_keys.py`. Verify internet connection. |
| **OOM during training** | Reduce batch size in `universal_trainer.py` or limit tickers. |
| **GDrive backup failed** | Re-authorise: `python backup_gdrive.py --setup` |

---

## License

MIT
