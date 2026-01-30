# AI Trading Server

Distributed model training server for 135+ stocks with zero-downtime deployment.

## Architecture

- **Data Fetcher**: Caches market data with rate limiting (2-hour TTL)
- **Model Trainer**: Dual-head LSTM for classification + regression
- **FastAPI Server**: REST API for model distribution and training control
- **Scheduler**: Daily automated retraining after market close

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-server.git
cd ai-trading-server
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Server (Optional)
Edit `server.py` to customize:
- Training schedule (default: 23:30 UTC)
- Port (default: 8000)
- Number of stocks (default: TOP_STOCKS[:20])

### 5. Run Server
```bash
python server.py
```

Server will be available at `http://0.0.0.0:8000`

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
