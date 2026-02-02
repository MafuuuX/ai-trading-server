"""
Distributed Training Server with FastAPI
Serves models, manages training, and enables zero-downtime updates
"""
import asyncio
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import logging
import base64
import secrets
import hmac
import psutil

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request, Response, Form
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd

from data_fetcher import CachedDataFetcher, TOP_STOCKS, get_live_prices
from trainer import ModelTrainer

# Sector mapping for Train Sector
SECTORS = {
    "Mega Cap": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    "Tech": ['GOOG', 'AVGO', 'ORCL', 'INTC', 'AMD', 'NFLX', 'CRM', 'ADBE', 'IBM', 'CSCO', 'QCOM', 'TXN'],
    "Finance": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'COIN', 'V', 'MA', 'AXP'],
    "Healthcare": ['UNH', 'JNJ', 'LLY', 'MRK', 'PFE', 'ABBV', 'TMO', 'AMGN'],
    "Energy": ['CVX', 'XOM', 'COP', 'MPC', 'PSX', 'VLO', 'HES', 'SLB'],
    "Consumer": ['MCD', 'SBUX', 'NKE', 'LULU', 'WMT', 'KO', 'PEP', 'HD'],
    "ETFs": ['SPY', 'QQQ', 'IWM', 'EEM', 'XLK', 'XLV', 'XLF', 'XLE']
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="AI Trading Server", version="1.0.0")

# UI
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Login auth for web UI
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change_this_secret")
SESSION_COOKIE = "ai_trading_session"


def _sign_session(username: str) -> str:
    payload = f"{username}"
    signature = hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    token = f"{payload}.{signature}"
    return base64.b64encode(token.encode("utf-8")).decode("utf-8")


def _verify_session(token: str) -> bool:
    try:
        decoded = base64.b64decode(token).decode("utf-8")
        payload, signature = decoded.rsplit(".", 1)
        expected = hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)
    except Exception:
        return False


def _is_logged_in(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return False
    return _verify_session(token)


@app.middleware("http")
async def ui_login_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/ui"):
        if not _is_logged_in(request):
            return RedirectResponse("/login")
    return await call_next(request)

# Global state
class ServerState:
    def __init__(self):
        self.models_dir = Path("./models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.models_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_models = {}  # ticker -> model_version
        self.training_status = {}  # ticker -> status
        self.training_queue = []  # list of tickers to process
        self.training_history = []  # completed trainings
        self.is_training = False
        self.current_training_ticker = None  # currently processing ticker
        self.training_progress = {}  # ticker -> 0-100
        self.training_start = {}  # ticker -> datetime
        self.training_metrics = {}  # ticker -> metrics
        self.training_history = []
        self.last_training = {}
        self.logs = []
        
        # Model version tracking
        self.model_versions_file = Path("./data/model_versions.json")
        self.model_versions_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_model_versions()

        # Live price cache for chart data
        self.live_prices_cache = {}  # ticker -> list of {price, timestamp}
        self.live_prices_max_points = 500  # Keep up to 500 price points per ticker
        self.chart_cache_file = Path("./data/chart_cache.json")
        self.chart_cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_chart_cache()
        
        self.scheduler = BackgroundScheduler()
        self.fetcher = CachedDataFetcher()
        self.trainer = ModelTrainer()
        
        # Track startup time for uptime calculation
        self.startup_time = datetime.now()

    def _load_model_versions(self):
        """Load model versions from disk and backfill from existing models"""
        try:
            if self.model_versions_file.exists():
                with open(self.model_versions_file, 'r') as f:
                    data = json.load(f)
                    self.active_models = data.get('versions', {})
        except Exception as e:
            logger.warning(f"Could not load model versions: {e}")
            self.active_models = {}

        # Backfill for existing model files
        try:
            for model_file in self.models_dir.glob("*_model.h5"):
                ticker = model_file.stem.replace("_model", "")
                if ticker not in self.active_models:
                    self.active_models[ticker] = "v1"
        except Exception as e:
            logger.warning(f"Could not backfill model versions: {e}")

        self.save_model_versions()

    def save_model_versions(self):
        """Save model versions to disk"""
        try:
            with open(self.model_versions_file, 'w') as f:
                json.dump({
                    'versions': self.active_models,
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Could not save model versions: {e}")
    
    def _load_chart_cache(self):
        """Load cached chart data from disk"""
        try:
            if self.chart_cache_file.exists():
                with open(self.chart_cache_file, 'r') as f:
                    data = json.load(f)
                    self.live_prices_cache = data.get('live_prices', {})
                    logger.info(f"Loaded chart cache with {len(self.live_prices_cache)} tickers")
        except Exception as e:
            logger.warning(f"Could not load chart cache: {e}")
            self.live_prices_cache = {}
    
    def save_chart_cache(self):
        """Save chart data cache to disk"""
        try:
            with open(self.chart_cache_file, 'w') as f:
                json.dump({
                    'live_prices': self.live_prices_cache,
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Could not save chart cache: {e}")
    
    def add_live_price(self, ticker: str, price: float):
        """Add a live price point to the cache"""
        if ticker not in self.live_prices_cache:
            self.live_prices_cache[ticker] = []
        
        self.live_prices_cache[ticker].append({
            'price': price,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit cache size
        if len(self.live_prices_cache[ticker]) > self.live_prices_max_points:
            self.live_prices_cache[ticker] = self.live_prices_cache[ticker][-self.live_prices_max_points:]

state = ServerState()


class InMemoryLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        state.logs.append({"time": datetime.now().isoformat(), "message": msg})
        if len(state.logs) > 200:
            state.logs = state.logs[-200:]


log_handler = InMemoryLogHandler()
log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(log_handler)

# ============================================================================
# DATA MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    ticker: str
    signal: str  # UP, DOWN, NEUTRAL
    confidence: float
    predicted_change: float
    timestamp: str

class TrainingStatus(BaseModel):
    ticker: str
    status: str  # idle, training, completed, failed
    progress: float  # 0-100
    last_trained: Optional[str]
    next_training: Optional[str]

class ModelInfo(BaseModel):
    ticker: str
    version: str
    trained_at: str
    file_hash: str
    file_size: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    active_models: int
    training_queue: int
    uptime_seconds: float

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Server health check"""
    uptime = (datetime.now() - state.startup_time).total_seconds()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_models": len(state.active_models),
        "training_queue": sum(1 for s in state.training_status.values() if s == "training"),
        "uptime_seconds": uptime
    }


@app.get("/api/metrics")
async def get_metrics():
    """System metrics"""
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_percent": mem.percent,
        "ram_used": mem.used,
        "ram_total": mem.total
    }


# ============================================================================
# CHART DATA CACHE ENDPOINTS
# ============================================================================

@app.get("/api/chart-cache")
async def get_chart_cache():
    """Get cached live price data for all tickers - used by app on startup"""
    return {
        "prices": state.live_prices_cache,
        "last_updated": datetime.now().isoformat(),
        "max_points": state.live_prices_max_points
    }


@app.get("/api/chart-cache/{ticker}")
async def get_chart_cache_ticker(ticker: str):
    """Get cached live price data for a specific ticker"""
    ticker = ticker.upper()
    if ticker not in state.live_prices_cache:
        return {"prices": [], "ticker": ticker}
    return {
        "prices": state.live_prices_cache[ticker],
        "ticker": ticker,
        "count": len(state.live_prices_cache[ticker])
    }


@app.post("/api/chart-cache/{ticker}")
async def add_chart_price(ticker: str, price: float):
    """Add a live price point to the cache"""
    ticker = ticker.upper()
    state.add_live_price(ticker, price)
    return {"status": "ok", "ticker": ticker, "price": price}


@app.post("/api/chart-cache/batch")
async def add_chart_prices_batch(prices: Dict[str, float]):
    """Add multiple live price points at once"""
    try:
        added = 0
        for ticker, price in prices.items():
            try:
                if price is not None and price > 0:
                    state.add_live_price(ticker.upper(), float(price))
                    added += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid price for {ticker}: {price} - {e}")
                continue
        
        # Save cache periodically (every 10 updates or so)
        if sum(len(v) for v in state.live_prices_cache.values()) % 10 == 0:
            state.save_chart_cache()
        
        return {"status": "ok", "count": len(prices), "added": added}
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/chart-cache")
async def clear_chart_cache():
    """Clear all cached chart data"""
    state.live_prices_cache = {}
    state.save_chart_cache()
    return {"status": "cleared"}


@app.get("/api/sectors")
async def get_sectors():
    """List available sectors"""
    return {"sectors": list(SECTORS.keys())}


@app.get("/api/performance")
async def get_performance():
    """Latest model performance per ticker"""
    return {"performance": state.training_metrics}


@app.get("/api/training-history")
async def get_training_history():
    """Training history timeline"""
    return {"history": state.training_history[-100:]}


@app.get("/api/logs")
async def get_logs():
    """Recent server logs"""
    return {"logs": state.logs}


@app.get("/api/queue")
async def get_queue():
    """Current training queue with proper counting"""
    # Get pending queue (not yet processed)
    pending = [t for t in state.training_queue if t != state.current_training_ticker]
    
    # Count currently training
    training_count = 1 if state.current_training_ticker else 0
    
    avg_duration = _average_training_duration()
    eta_seconds = None
    total_items = len(pending) + training_count
    if avg_duration and total_items > 0:
        eta_seconds = int(avg_duration * total_items)
    
    return {
        "queue": pending,  # Return pending queue (not including current)
        "current": state.current_training_ticker,  # Currently training
        "count": total_items,  # Total items (current + pending)
        "eta_seconds": eta_seconds
    }

@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    models = []
    for model_file in state.models_dir.glob("*_model.h5"):
        ticker = model_file.stem.replace("_model", "")
        file_stat = model_file.stat()
        
        # Calculate file hash
        with open(model_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        models.append({
            "ticker": ticker,
            "version": state.active_models.get(ticker, "unknown"),
            "trained_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "file_hash": file_hash,
            "file_size": file_stat.st_size
        })
    
    return models


@app.post("/api/models/{ticker}/rollback")
async def rollback_model(ticker: str):
    """Rollback to the latest backup for a ticker"""
    ticker = ticker.upper()
    ticker_backup_dir = state.backup_dir / ticker
    if not ticker_backup_dir.exists():
        raise HTTPException(status_code=404, detail="No backups available")
    
    backups = sorted(ticker_backup_dir.glob(f"{ticker}_model_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    scaler_backups = sorted(ticker_backup_dir.glob(f"{ticker}_scaler_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups or not scaler_backups:
        raise HTTPException(status_code=404, detail="No valid backup pair found")
    
    # Restore latest
    model_path = state.models_dir / f"{ticker}_model.h5"
    scaler_path = state.models_dir / f"{ticker}_scaler.pkl"
    backups[0].replace(model_path)
    scaler_backups[0].replace(scaler_path)
    
    if ticker not in state.active_models:
        state.active_models[ticker] = "v1"
        state.save_model_versions()
    logger.info(f"Rolled back {ticker} to latest backup")
    return {"status": "rolled_back", "ticker": ticker}

@app.get("/api/training-status", response_model=Dict[str, TrainingStatus])
async def training_status():
    """Get training status for all stocks - cached for 2 seconds"""
    # Simple cache to avoid re-computing this every request
    current_time = time.time()
    if hasattr(state, '_status_cache') and hasattr(state, '_status_cache_time'):
        if current_time - state._status_cache_time < 2:  # Cache for 2 seconds
            return state._status_cache
    
    status_dict = {}
    next_run = None
    job = state.scheduler.get_job('daily_training') if state.scheduler else None
    if job and job.next_run_time:
        next_run = job.next_run_time.isoformat()
    
    # Only return status for stocks that have been queued or trained
    # This is much faster than iterating all 135 stocks every time
    active_tickers = set(state.training_status.keys()) | set(state.last_training.keys())
    
    for ticker in active_tickers:
        status_dict[ticker] = {
            "ticker": ticker,
            "status": state.training_status.get(ticker, "idle"),
            "progress": state.training_progress.get(ticker, 0.0),
            "last_trained": state.last_training.get(ticker),
            "next_training": next_run
        }
    
    # Cache the result
    state._status_cache = status_dict
    state._status_cache_time = current_time
    
    return status_dict

# ============================================================================
# MODEL DOWNLOAD ENDPOINTS
# ============================================================================

@app.get("/api/models/{ticker}/download")
async def download_model(ticker: str):
    """Download model and scaler for a stock"""
    ticker = ticker.upper()
    
    model_path = state.models_dir / f"{ticker}_model.h5"
    scaler_path = state.models_dir / f"{ticker}_scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        raise HTTPException(status_code=404, detail=f"Model for {ticker} not found")
    
    # Create zip file with both model and scaler
    import shutil
    zip_path = state.models_dir / f"{ticker}_model.zip"
    
    # Create temporary directory
    temp_dir = state.models_dir / f"{ticker}_temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Copy files
    shutil.copy(model_path, temp_dir / f"{ticker}_model.h5")
    shutil.copy(scaler_path, temp_dir / f"{ticker}_scaler.pkl")
    
    # Create zip
    shutil.make_archive(
        str(zip_path.parent / f"{ticker}_model"),
        'zip',
        str(temp_dir)
    )
    
    # Clean up temp
    shutil.rmtree(temp_dir)
    
    return FileResponse(
        path=zip_path,
        filename=f"{ticker}_model.zip",
        media_type="application/zip"
    )

@app.get("/api/models/{ticker}/hash")
async def get_model_hash(ticker: str):
    """Get hash of model file for change detection"""
    ticker = ticker.upper()
    model_path = state.models_dir / f"{ticker}_model.h5"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model for {ticker} not found")
    
    with open(model_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    return {"ticker": ticker, "hash": file_hash}

# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

@app.post("/api/train/{ticker}")
async def train_stock(ticker: str, background_tasks: BackgroundTasks):
    """Start training for a specific stock"""
    ticker = ticker.upper()
    
    if state.training_status.get(ticker) in ("training", "queued"):
        raise HTTPException(status_code=409, detail=f"{ticker} already training")
    
    state.training_status[ticker] = "queued"
    if ticker not in state.training_queue:
        state.training_queue.append(ticker)
    background_tasks.add_task(process_training_queue)
    
    return {"status": "queued", "ticker": ticker}

def train_stock_task(ticker: str):
    """Background task to train a stock"""
    try:
        logger.info(f"Starting training for {ticker}...")
        state.training_start[ticker] = datetime.now()
        state.training_progress[ticker] = 0.0
        
        # Fetch data
        df = state.fetcher.fetch_historical_data(ticker)
        if df is None:
            state.training_status[ticker] = "failed"
            state.training_progress[ticker] = 0.0
            reason = getattr(state.fetcher, "last_error", None)
            if reason:
                logger.error(f"Insufficient data for {ticker}: {reason}")
            else:
                logger.error(f"Insufficient data for {ticker}: df is None")
            return
        if len(df) < 100:
            state.training_status[ticker] = "failed"
            state.training_progress[ticker] = 0.0
            logger.error(f"Insufficient data for {ticker}: {len(df)} rows")
            return
        
        # Progress callback
        def on_progress(epoch, total):
            state.training_progress[ticker] = round((epoch / total) * 100, 1)

        # Train model with timeout safety
        try:
            result = state.trainer.train_and_validate(df, ticker, progress_callback=on_progress)
        except Exception as train_err:
            logger.error(f"Training error for {ticker}: {train_err}")
            state.training_status[ticker] = "failed"
            state.training_progress[ticker] = 0.0
            return
        
        if result:
            state.training_status[ticker] = "completed"
            state.training_progress[ticker] = 100.0  # Set to 100% when completed
            state.last_training[ticker] = datetime.now().isoformat()
            # Increment model version
            current_version = state.active_models.get(ticker, "v0")
            try:
                current_num = int(str(current_version).lstrip("v"))
            except ValueError:
                current_num = 0
            new_version = f"v{current_num + 1}"
            state.active_models[ticker] = new_version
            state.save_model_versions()
            state.training_metrics[ticker] = {
                "class_accuracy": result.get("class_accuracy"),
                "reg_mae": result.get("reg_mae"),
                "trained_at": result.get("trained_at")
            }
            duration = (datetime.now() - state.training_start[ticker]).total_seconds()
            state.training_history.append({
                "ticker": ticker,
                "status": "completed",
                "duration_seconds": duration,
                "trained_at": state.last_training[ticker],
                "class_accuracy": result.get("class_accuracy"),
                "reg_mae": result.get("reg_mae")
            })
            logger.info(f"✅ Completed training for {ticker} in {duration:.0f}s")
        else:
            state.training_status[ticker] = "failed"
            state.training_progress[ticker] = 0.0
            duration = (datetime.now() - state.training_start[ticker]).total_seconds()
            state.training_history.append({
                "ticker": ticker,
                "status": "failed",
                "duration_seconds": duration,
                "trained_at": datetime.now().isoformat()
            })
            logger.error(f"Training failed for {ticker}")
            
    except Exception as e:
        import traceback
        state.training_status[ticker] = "failed"
        duration = (datetime.now() - state.training_start[ticker]).total_seconds() if ticker in state.training_start else None
        state.training_history.append({
            "ticker": ticker,
            "status": "failed",
            "duration_seconds": duration,
            "trained_at": datetime.now().isoformat(),
            "error": str(e)
        })
        # Log with full traceback
        logger.error(f"Error training {ticker}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


def process_training_queue():
    """Process queued trainings sequentially with proper state tracking"""
    if state.is_training:
        return  # Already processing
    
    state.is_training = True
    try:
        while state.training_queue:
            ticker = state.training_queue.pop(0)
            state.current_training_ticker = ticker  # Track what we're training
            state.training_status[ticker] = "training"
            try:
                train_stock_task(ticker)
            except Exception as e:
                logger.error(f"Error training {ticker}: {e}")
                state.training_status[ticker] = "failed"
            finally:
                state.current_training_ticker = None  # Clear when done
    except Exception as e:
        logger.error(f"Queue processing error: {e}")
    finally:
        state.is_training = False  # Always reset flag

@app.post("/api/train-batch")
async def train_batch(tickers: List[str] = None, background_tasks: BackgroundTasks = None):
    """Start training for multiple stocks"""
    if tickers is None:
        tickers = TOP_STOCKS[:10]  # Default first 10
    
    for ticker in tickers:
        if state.training_status.get(ticker) not in ("training", "queued"):
            state.training_status[ticker] = "queued"
            if ticker not in state.training_queue:
                state.training_queue.append(ticker)
    if background_tasks:
        background_tasks.add_task(process_training_queue)
    
    return {"status": "batch queued", "count": len(tickers)}


@app.post("/api/train-all")
async def train_all(background_tasks: BackgroundTasks):
    """Queue training for all stocks"""
    for ticker in TOP_STOCKS:
        if state.training_status.get(ticker) not in ("training", "queued"):
            state.training_status[ticker] = "queued"
            if ticker not in state.training_queue:
                state.training_queue.append(ticker)
    background_tasks.add_task(process_training_queue)
    return {"status": "all queued", "count": len(TOP_STOCKS)}


@app.post("/api/train-sector/{sector}")
async def train_sector(sector: str, background_tasks: BackgroundTasks):
    """Queue training for a sector"""
    if sector not in SECTORS:
        raise HTTPException(status_code=404, detail="Sector not found")
    tickers = SECTORS[sector]
    for ticker in tickers:
        if state.training_status.get(ticker) not in ("training", "queued"):
            state.training_status[ticker] = "queued"
            if ticker not in state.training_queue:
                state.training_queue.append(ticker)
    background_tasks.add_task(process_training_queue)
    return {"status": "sector queued", "sector": sector, "count": len(tickers)}


def _average_training_duration() -> Optional[float]:
    durations = [h["duration_seconds"] for h in state.training_history if h.get("duration_seconds")]
    if not durations:
        return None
    return sum(durations) / len(durations)

# ============================================================================
# SCHEDULER SETUP
# ============================================================================

def schedule_daily_training():
    """Schedule daily model training after market close"""
    # Run daily at 23:30 UTC (after US market close)
    state.scheduler.add_job(
        train_end_of_day,
        'cron',
        hour=23,
        minute=30,
        id='daily_training'
    )
    logger.info("Daily training scheduled at 23:30 UTC")

def schedule_live_price_collection():
    """Schedule periodic live price collection for chart cache"""
    # Collect live prices every 3 seconds
    state.scheduler.add_job(
        collect_live_prices,
        'interval',
        seconds=3,
        id='live_price_collection'
    )
    logger.info("Live price collection scheduled (every 3s)")

def collect_live_prices():
    """Collect live prices for all tickers and cache them"""
    try:
        prices = get_live_prices(TOP_STOCKS)
        
        added = 0
        for ticker, price in prices.items():
            if price is not None and price > 0:
                try:
                    state.add_live_price(ticker.upper(), float(price))
                    added += 1
                except (ValueError, TypeError):
                    continue
        
        # Save cache periodically (every 20 collections = 1 minute)
        run_count = getattr(state, '_price_collection_count', 0) + 1
        state._price_collection_count = run_count
        
        if run_count % 20 == 0:
            state.save_chart_cache()
            logger.info(f"Chart cache saved: {added}/{len(TOP_STOCKS)} prices collected")
    except Exception as e:
        logger.error(f"Error collecting live prices: {e}")

def train_end_of_day():
    """Train all models at end of day"""
    logger.info(f"Starting end-of-day training for {len(TOP_STOCKS)} stocks...")
    
    for ticker in TOP_STOCKS:  # Train all stocks
        if state.training_status.get(ticker) not in ("training", "queued"):
            state.training_status[ticker] = "queued"
            if ticker not in state.training_queue:
                state.training_queue.append(ticker)
    
    # Process queue in separate thread to avoid blocking scheduler
    import threading
    training_thread = threading.Thread(target=process_training_queue, daemon=False)
    training_thread.start()
    logger.info(f"Training queue populated with {len(state.training_queue)} stocks")

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    logger.info("🚀 Server starting up...")
    
    # Start scheduler
    if not state.scheduler.running:
        state.scheduler.start()
        schedule_daily_training()
    
    logger.info("✅ Server initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Server shutting down...")
    
    # Save chart cache before shutdown
    state.save_chart_cache()
    logger.info("💾 Chart cache saved")
    
    if state.scheduler.running:
        state.scheduler.shutdown()

# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Trading Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/login")
async def login_page(request: Request):
    """Login page"""
    if _is_logged_in(request):
        return RedirectResponse("/ui")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login"""
    if secrets.compare_digest(username, ADMIN_USER) and secrets.compare_digest(password, ADMIN_PASS):
        response = RedirectResponse("/ui", status_code=302)
        token = _sign_session(username)
        response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
        return response
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session"""
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/ui")
async def ui_dashboard(request: Request):
    """Simple web GUI dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Server configuration
    host = "0.0.0.0"  # Listen on all interfaces
    port = 8000
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
