# AI Trading Server

Enterprise-grade distributed model training server for 135+ stocks with advanced technical analysis and dual-head LSTM architecture.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technical Indicators](#technical-indicators)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

---

## Features

✅ **Dual-Head LSTM Model** - Simultaneous classification and regression predictions  
✅ **7 Technical Indicators** - All implemented with pure pandas (no external dependencies)  
✅ **Intelligent Data Fallback** - YFinance → Stooq (Polish markets) seamless switching  
✅ **Smart Caching** - 2-hour TTL with automatic validation  
✅ **Zero-Downtime Training** - Background jobs don't block API  
✅ **FastAPI REST Server** - Modern async Python web framework  
✅ **Systemd Integration** - Production-ready service configuration  
✅ **Comprehensive Logging** - Full tracebacks and performance metrics  

---

## Architecture

### Core Components

**Data Fetcher** (`data_fetcher.py`)
- Multi-source data strategy: YFinance (primary) → Stooq (fallback)
- Intelligent timeout handling (3-5 seconds)
- Polish column mapping: `Data→Date`, `Otwarcie→Open`, `Zamkniecie→Close`, etc.
- Pandas-based dataframe validation

**Model Trainer** (`trainer.py`)
- Dual-head LSTM: 7 technical indicators → [Classification, Regression] outputs
- 60-day lookback window
- MinMax scaling (0-1 normalization)
- Early stopping with validation monitoring
- 135+ stock ensemble training

**FastAPI Server** (`server.py`)
- Async request handling
- JWT-based authentication
- Model download/sync endpoints
- Training progress tracking
- HTML dashboard

**Scheduler**
- Automatic daily retraining (23:30 UTC)
- Cron-based execution
- Error recovery with exponential backoff

---

## Technical Indicators

All indicators implemented with **pandas only** - no external `ta` library dependency.

### 1. RSI (Relative Strength Index) - 14-period
```
Momentum oscillator measuring speed/magnitude of price changes
Range: 0-100 (>70 overbought, <30 oversold)
```

### 2. MACD (Moving Average Convergence Divergence)
```
Trend-following momentum indicator
EMA(12) - EMA(26) = trend signal
```

### 3. Bollinger Bands
```
Volatility bands around 20-day SMA
Upper = SMA + 2×StdDev
Lower = SMA - 2×StdDev
```

### 4. ATR (Average True Range) - 14-period
```
Volatility measure (price movement amplitude)
Max(High-Low, |High-Close(prev)|, |Low-Close(prev)|)
```

### 5. SMA (Simple Moving Average)
```
20-day and 50-day trend indicators
```

### 6. Returns & Volatility
```
Daily returns: Close(t) / Close(t-1) - 1
Rolling volatility: 20-day standard deviation
```

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

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pandas ≥ 2.0.0 - Data manipulation
- numpy ≥ 1.24.0 - Numerical computing
- tensorflow ≥ 2.13.0 - Deep learning
- scikit-learn ≥ 1.3.0 - Machine learning utilities
- yfinance ≥ 0.2.28 - Market data
- requests ≥ 2.31.0 - HTTP library

### 4. (Optional) Install Systemd Service
```bash
sudo cp ai-trading-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-server
sudo systemctl start ai-trading-server

# Monitor logs
sudo journalctl -u ai-trading-server -f
```

---

## Configuration

### Environment Variables
```bash
export ADMIN_USER="admin"
export ADMIN_PASS="your_secure_password"
export SESSION_SECRET="your_session_secret_key"
export TRAINING_HOUR=23  # UTC hour for daily retraining
export TRAINING_MINUTE=30
```

### server.py Settings
```python
TOP_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',   # Mega Cap
    'NVDA', 'TSLA', 'AMD', 'NFLX', 'GOOG',    # Large Cap
    # ... plus 125 more stocks
]
MODEL_DIR = './models'
LOOKBACK = 60  # days
BATCH_SIZE = 32
EPOCHS = 50
```

---

## Usage

### Start Server
```bash
python server.py
```
Server runs on `http://0.0.0.0:8000`

### Access Web Dashboard
```
http://localhost:8000
```
- Login with ADMIN_USER / ADMIN_PASS
- View training progress
- Download models
- Monitor performance metrics

### Trigger Manual Training
```bash
# Via API
curl -X POST http://localhost:8000/api/train \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stocks": ["AAPL", "MSFT", "GOOGL"]}'

# Or via web dashboard
# Click "Train Mega Cap" or "Train All 135"
```

---

## API Documentation

### Health Check
```
GET /api/health
Response: {"status": "operational", "timestamp": "2026-01-30T21:45:00"}
```

### Get Available Models
```
GET /api/models
Response: [
  {
    "ticker": "AAPL",
    "trained": "2026-01-30T15:30:00",
    "classifier_file": "AAPL_classifier.keras",
    "regressor_file": "AAPL_regressor.keras"
  },
  ...
]
```

### Download Model Bundle
```
GET /api/models/download
Response: ZIP file containing all models + scalers
```

### Training Status
```
GET /api/training/status
Response: {
  "active_jobs": 3,
  "queue_size": 12,
  "current": "AAPL (Step 45/50)",
  "progress": 0.90
}
```

---

## API Endpoints

### Health & Info
- `GET /api/health` - Server status
- `GET /api/models` - List available models
- `GET /api/training-status` - Training progress for all stocks

### Model Management
- `GET /api/models/{ticker}/download` - Download model + scaler as ZIP
- `GET /api/models/{ticker}/hash` - Get model file hash (for change detection)

### Training
- `POST /api/train/{ticker}` - Train specific stock
- `POST /api/train-batch` - Train multiple stocks

## Stock Coverage

**135 stocks** across:
- Mega Cap: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- Large Cap Tech: GOOG, AVGO, ORCL, INTC, AMD, NFLX, CRM, ADBE, etc.
- Finance: JPM, BAC, WFC, GS, MS, BLK, SCHW, COIN, V, MA, etc.
- Healthcare: UNH, JNJ, LLY, MRK, PFE, ABBV, TMO, AMGN, etc.
- Energy: CVX, XOM, COP, MPC, PSX, VLO, HES, SLB, etc.
- Consumer: MCD, SBUX, DKNG, NKE, LULU, ULTA, GPS, WMT, etc.
- ETFs: SPY, QQQ, IWM, EEM, FXI, BABA, IGV, XLK, etc.

## Training Schedule

- **Mo-Fr 23:30 UTC**: Fetch new market data from yfinance
- **Tu-Sa 01:00 UTC**: Start model training (~4-6 hours on i5-6400)
- **After completion**: Automatically promote to production

## Model Format

- **Classification**: 3-class (DOWN/NEUTRAL/UP)
- **Regression**: Percentage change prediction
- **Lookback**: 30 days
- **Features**: 12 technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA, etc.)

## Client Integration

### Check for Model Updates
```python
import requests
import hashlib

def check_model_update(ticker):
    response = requests.get(f"http://server:8000/api/models/{ticker}/hash")
    return response.json()['hash']

def download_model(ticker):
    response = requests.get(f"http://server:8000/api/models/{ticker}/download")
    with open(f"{ticker}_model.zip", 'wb') as f:
        f.write(response.content)
```

## Performance

- **i5-6400 (4c/4t, 24GB RAM)**: ~4-6 hours for 135 stocks
- **Memory per model**: ~50MB (model + scaler)
- **Total model cache**: ~7GB for all 135 stocks
- **API response time**: <100ms for model download

## Monitoring

Check server health:
```bash
curl http://192.168.2.96:8000/api/health
```

View training status:
```bash
curl http://192.168.2.96:8000/api/training-status
```

List available models:
```bash
curl http://192.168.2.96:8000/api/models
```

## Troubleshooting

**"Too Many Requests"**: yfinance rate limiting
- Solution: Data fetcher has built-in 2-hour cache and 0.5s rate limiting

**OOM (Out of Memory)**: Training 135 stocks simultaneously
- Solution: Reduce batch size in trainer.py or train in smaller batches

**Training takes too long**: i5-6400 is baseline hardware
- Solution: Upgrade CPU or train fewer stocks (~40-50 optimal for this CPU)

## License

MIT
