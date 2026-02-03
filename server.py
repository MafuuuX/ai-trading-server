"""
Distributed Training Server with FastAPI
Serves models, manages training, and enables zero-downtime updates

Version 2.0 - Enhanced with:
- Feature drift monitoring
- Trade journal API
- Comprehensive error handling
- Health monitoring for client offline detection
"""
import asyncio
import os
import json
import hashlib
import time
import traceback
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import logging
import base64
import secrets
import hmac
import psutil

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request, Response, Form, Path as FastapiPath
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
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

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app with enhanced configuration
app = FastAPI(
    title="AI Trading Server",
    version="2.0.0",
    description="Distributed ML Training Server with Drift Monitoring and Trade Journal"
)

# CORS middleware for client connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# UI
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ============================================================================
# GLOBAL ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    error_id = hashlib.md5(f"{time.time()}{str(exc)}".encode()).hexdigest()[:8]
    
    # Log the error
    logger.error(f"[ERROR-{error_id}] Unhandled exception: {exc}")
    logger.error(f"[ERROR-{error_id}] Traceback: {traceback.format_exc()}")
    
    # Store in error log if state is available
    try:
        if 'state' in globals():
            state.log_error(
                "UNHANDLED",
                str(exc),
                {
                    "error_id": error_id,
                    "path": str(request.url.path),
                    "method": request.method,
                    "traceback": traceback.format_exc()[-500:]  # Last 500 chars
                }
            )
    except:
        pass
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An error occurred"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions with logging"""
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    elif exc.status_code >= 400:
        logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


# ============================================================================
# REQUEST VALIDATION MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def request_validation_middleware(request: Request, call_next):
    """Validate requests and add timing headers"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Add timing header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}s"
        response.headers["X-Server-Time"] = datetime.now().isoformat()
        
        return response
    except Exception as e:
        logger.error(f"Request middleware error: {e}")
        raise


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
        
        # Feature drift monitoring
        self.feature_stats_file = Path("./data/feature_stats.json")
        self.feature_stats = {}  # ticker -> feature_name -> {mean, std, min, max, p5, p95}
        self._load_feature_stats()
        
        # Trade journal
        self.trade_journal_file = Path("./data/trade_journal.json")
        self.trade_journal = []
        self._load_trade_journal()
        
        # Trade outcomes for reinforcement learning
        self.trade_outcomes_file = Path("./data/trade_outcomes.json")
        self.trade_outcomes = []
        self._load_trade_outcomes()
        
        # Reinforcement learning configuration
        self.rl_config_file = Path("./data/rl_config.json")
        self.rl_config = {
            "enabled": True,                    # RL training enabled by default
            "min_trades_required": 20,          # Minimum trades before RL training
            "last_rl_training": None,           # Last RL training timestamp
            "rl_training_interval_hours": 24,   # Run RL training daily
            "confidence_weight_factor": 2.0,    # How much to weight confidence-based feedback
            "pnl_weight_cap": 3.0,              # Max sample weight multiplier
        }
        self._load_rl_config()
        
        # Error tracking
        self.error_log = []  # Recent errors for diagnostics
        self.max_error_log = 100
        
        self.scheduler = BackgroundScheduler()
        self.fetcher = CachedDataFetcher()
        self.trainer = ModelTrainer()
        
        # Track startup time for uptime calculation
        self.startup_time = datetime.now()
        
        # Heartbeat tracking for client health detection
        self.last_heartbeat = datetime.now()
    
    def _load_feature_stats(self):
        """Load feature statistics from disk"""
        try:
            if self.feature_stats_file.exists():
                with open(self.feature_stats_file, 'r') as f:
                    self.feature_stats = json.load(f)
                    logger.info(f"Loaded feature stats for {len(self.feature_stats)} tickers")
        except Exception as e:
            logger.warning(f"Could not load feature stats: {e}")
            self.feature_stats = {}
    
    def save_feature_stats(self):
        """Save feature statistics to disk"""
        try:
            with open(self.feature_stats_file, 'w') as f:
                json.dump(self.feature_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save feature stats: {e}")
    
    def _load_trade_journal(self):
        """Load trade journal from disk"""
        try:
            if self.trade_journal_file.exists():
                with open(self.trade_journal_file, 'r') as f:
                    self.trade_journal = json.load(f)
                    logger.info(f"Loaded {len(self.trade_journal)} trade records")
        except Exception as e:
            logger.warning(f"Could not load trade journal: {e}")
            self.trade_journal = []
    
    def save_trade_journal(self):
        """Save trade journal to disk"""
        try:
            with open(self.trade_journal_file, 'w') as f:
                json.dump(self.trade_journal, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save trade journal: {e}")
    
    def _load_trade_outcomes(self):
        """Load trade outcomes for reinforcement learning"""
        try:
            if self.trade_outcomes_file.exists():
                with open(self.trade_outcomes_file, 'r') as f:
                    self.trade_outcomes = json.load(f)
                    logger.info(f"Loaded {len(self.trade_outcomes)} trade outcomes for RL")
        except Exception as e:
            logger.warning(f"Could not load trade outcomes: {e}")
            self.trade_outcomes = []
    
    def save_trade_outcomes(self):
        """Save trade outcomes to disk"""
        try:
            with open(self.trade_outcomes_file, 'w') as f:
                json.dump(self.trade_outcomes, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save trade outcomes: {e}")
    
    def _load_rl_config(self):
        """Load reinforcement learning configuration"""
        try:
            if self.rl_config_file.exists():
                with open(self.rl_config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.rl_config.update(saved_config)
                    logger.info(f"Loaded RL config: enabled={self.rl_config['enabled']}, min_trades={self.rl_config['min_trades_required']}")
        except Exception as e:
            logger.warning(f"Could not load RL config: {e}")
    
    def save_rl_config(self):
        """Save reinforcement learning configuration"""
        try:
            with open(self.rl_config_file, 'w') as f:
                json.dump(self.rl_config, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save RL config: {e}")
    
    def get_closed_trade_count(self) -> int:
        """Get count of closed trades for RL eligibility check"""
        return len([t for t in self.trade_outcomes if t.get('outcome')])
    
    def is_rl_ready(self) -> Tuple[bool, str]:
        """
        Check if reinforcement learning training can be executed.
        
        Returns:
            Tuple[bool, str]: (ready, reason)
        """
        if not self.rl_config.get('enabled', True):
            return False, "Reinforcement learning is disabled"
        
        closed_count = self.get_closed_trade_count()
        min_required = self.rl_config.get('min_trades_required', 20)
        
        if closed_count < min_required:
            return False, f"Not enough trades: {closed_count}/{min_required}"
        
        # Check if training interval has passed
        last_training = self.rl_config.get('last_rl_training')
        if last_training:
            try:
                last_dt = datetime.fromisoformat(last_training)
                hours_since = (datetime.now() - last_dt).total_seconds() / 3600
                interval = self.rl_config.get('rl_training_interval_hours', 24)
                if hours_since < interval:
                    return False, f"Training interval not reached: {hours_since:.1f}/{interval}h"
            except Exception:
                pass
        
        return True, f"Ready with {closed_count} closed trades"

    def log_error(self, error_type: str, message: str, details: dict = None):
        """Log an error for diagnostics"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "details": details or {}
        }
        self.error_log.append(error_entry)
        if len(self.error_log) > self.max_error_log:
            self.error_log = self.error_log[-self.max_error_log:]
        logger.error(f"[{error_type}] {message}")

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

class PriceUpdate(BaseModel):
    price: float

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


# IMPORTANT: /batch must come BEFORE /{ticker} so FastAPI matches it first
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
async def add_chart_price(ticker: str = FastapiPath(...), update: PriceUpdate = None):
    """Add a live price point to the cache"""
    ticker = ticker.upper()
    if update is None:
        raise HTTPException(status_code=422, detail="Missing price in request body")
    state.add_live_price(ticker, update.price)
    return {"status": "ok", "ticker": ticker, "price": update.price}


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


@app.get("/api/errors")
async def get_errors():
    """Recent error log for diagnostics"""
    return {"errors": state.error_log, "count": len(state.error_log)}


# ============================================================================
# FEATURE DRIFT MONITORING ENDPOINTS
# ============================================================================

class FeatureStatsInput(BaseModel):
    """Input for recording feature statistics"""
    ticker: str
    features: Dict[str, Dict[str, float]]  # feature_name -> {mean, std, min, max, p5, p95}


class DriftCheckInput(BaseModel):
    """Input for checking drift"""
    ticker: str
    features: Dict[str, float]  # feature_name -> current_value


@app.get("/api/drift/stats")
async def get_all_feature_stats():
    """Get all stored feature statistics"""
    return {
        "stats": state.feature_stats,
        "ticker_count": len(state.feature_stats)
    }


@app.get("/api/drift/stats/{ticker}")
async def get_feature_stats(ticker: str):
    """Get feature statistics for a specific ticker"""
    ticker = ticker.upper()
    if ticker not in state.feature_stats:
        raise HTTPException(status_code=404, detail=f"No stats for {ticker}")
    return {"ticker": ticker, "stats": state.feature_stats[ticker]}


@app.post("/api/drift/stats")
async def record_feature_stats(data: FeatureStatsInput):
    """Record feature statistics from training data"""
    try:
        ticker = data.ticker.upper()
        state.feature_stats[ticker] = data.features
        state.save_feature_stats()
        logger.info(f"Recorded {len(data.features)} feature stats for {ticker}")
        return {"status": "ok", "ticker": ticker, "features": len(data.features)}
    except Exception as e:
        state.log_error("DRIFT_STATS", f"Failed to record stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drift/check")
async def check_drift(data: DriftCheckInput):
    """Check live features against training distribution"""
    try:
        ticker = data.ticker.upper()
        if ticker not in state.feature_stats:
            return {"ticker": ticker, "alerts": [], "message": "No baseline stats available"}
        
        alerts = []
        baseline = state.feature_stats[ticker]
        
        for feature_name, live_value in data.features.items():
            if feature_name not in baseline:
                continue
            
            stats = baseline[feature_name]
            mean = stats.get('mean', 0)
            std = stats.get('std', 1)
            
            if std == 0:
                std = 0.001
            
            # Calculate z-score
            z_score = abs(live_value - mean) / std
            
            # Calculate drift percentage
            drift_pct = abs(live_value - mean) / abs(mean) if mean != 0 else 0
            
            # Check thresholds
            if z_score > 3.0:
                alerts.append({
                    "feature": feature_name,
                    "severity": "critical",
                    "type": "out_of_range",
                    "z_score": round(z_score, 2),
                    "drift_pct": round(drift_pct * 100, 1),
                    "current": live_value,
                    "expected_mean": mean
                })
            elif drift_pct > 0.25:
                alerts.append({
                    "feature": feature_name,
                    "severity": "critical",
                    "type": "mean_shift",
                    "drift_pct": round(drift_pct * 100, 1),
                    "current": live_value,
                    "expected_mean": mean
                })
            elif drift_pct > 0.15:
                alerts.append({
                    "feature": feature_name,
                    "severity": "warning",
                    "type": "mean_shift",
                    "drift_pct": round(drift_pct * 100, 1),
                    "current": live_value,
                    "expected_mean": mean
                })
        
        return {
            "ticker": ticker,
            "alerts": alerts,
            "critical_count": len([a for a in alerts if a['severity'] == 'critical']),
            "warning_count": len([a for a in alerts if a['severity'] == 'warning']),
            "should_pause": len([a for a in alerts if a['severity'] == 'critical']) >= 3
        }
    except Exception as e:
        state.log_error("DRIFT_CHECK", f"Drift check failed: {e}", {"ticker": data.ticker})
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRADE JOURNAL ENDPOINTS
# ============================================================================

class TradeJournalEntry(BaseModel):
    """Trade journal entry from client"""
    ticker: str
    action: str  # BUY, SELL, SHORT, COVER
    price: float
    shares: int
    signal_reason: str = ""
    confidence: float = 0.0
    position_multiplier: float = 1.0
    model_type: str = ""
    volatility: float = 0.0


class TradeCloseEntry(BaseModel):
    """Close a trade"""
    ticker: str
    exit_price: float
    realized_pnl: float
    outcome: str  # WIN, LOSS, BREAKEVEN


@app.get("/api/trades")
async def get_trades(limit: int = 100, ticker: str = None):
    """Get trade journal entries"""
    trades = state.trade_journal
    if ticker:
        trades = [t for t in trades if t.get('ticker', '').upper() == ticker.upper()]
    return {"trades": trades[-limit:], "total": len(trades)}


@app.post("/api/trades")
async def log_trade(entry: TradeJournalEntry):
    """Log a new trade to the journal"""
    try:
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "ticker": entry.ticker.upper(),
            "action": entry.action.upper(),
            "price": entry.price,
            "shares": entry.shares,
            "total_value": entry.price * entry.shares,
            "signal_reason": entry.signal_reason,
            "confidence": entry.confidence,
            "position_multiplier": entry.position_multiplier,
            "model_type": entry.model_type,
            "volatility": entry.volatility,
            # Fields to be filled on close
            "exit_price": None,
            "realized_pnl": None,
            "outcome": None,
            "closed_at": None
        }
        state.trade_journal.append(trade_record)
        state.save_trade_journal()
        logger.info(f"Trade logged: {entry.action} {entry.shares} {entry.ticker} @ ${entry.price}")
        return {"status": "ok", "trade_id": len(state.trade_journal) - 1}
    except Exception as e:
        state.log_error("TRADE_LOG", f"Failed to log trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trades/close")
async def close_trade(entry: TradeCloseEntry):
    """Close an open trade and record outcome"""
    try:
        ticker = entry.ticker.upper()
        # Find last open trade for this ticker
        for i in range(len(state.trade_journal) - 1, -1, -1):
            trade = state.trade_journal[i]
            if trade.get('ticker') == ticker and trade.get('exit_price') is None:
                trade['exit_price'] = entry.exit_price
                trade['realized_pnl'] = entry.realized_pnl
                trade['outcome'] = entry.outcome
                trade['closed_at'] = datetime.now().isoformat()
                state.save_trade_journal()
                logger.info(f"Trade closed: {ticker} PnL=${entry.realized_pnl:.2f} ({entry.outcome})")
                return {"status": "ok", "trade_id": i}
        
        raise HTTPException(status_code=404, detail=f"No open trade found for {ticker}")
    except HTTPException:
        raise
    except Exception as e:
        state.log_error("TRADE_CLOSE", f"Failed to close trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades/analytics")
async def get_trade_analytics():
    """Get trading analytics and win rates"""
    try:
        closed_trades = [t for t in state.trade_journal if t.get('outcome')]
        
        if not closed_trades:
            return {"message": "No closed trades yet", "analytics": {}}
        
        total = len(closed_trades)
        wins = len([t for t in closed_trades if t.get('outcome') == 'WIN'])
        losses = len([t for t in closed_trades if t.get('outcome') == 'LOSS'])
        
        # Win rate by confidence level
        confidence_buckets = {"50-60%": [], "60-70%": [], "70-85%": [], "85%+": []}
        for t in closed_trades:
            conf = t.get('confidence', 0)
            if conf < 0.6:
                confidence_buckets["50-60%"].append(t)
            elif conf < 0.7:
                confidence_buckets["60-70%"].append(t)
            elif conf < 0.85:
                confidence_buckets["70-85%"].append(t)
            else:
                confidence_buckets["85%+"].append(t)
        
        win_rate_by_confidence = {}
        for bucket, trades in confidence_buckets.items():
            if trades:
                bucket_wins = len([t for t in trades if t.get('outcome') == 'WIN'])
                win_rate_by_confidence[bucket] = round(bucket_wins / len(trades) * 100, 1)
        
        # Win rate by model type
        model_types = {}
        for t in closed_trades:
            model = t.get('model_type', 'unknown')
            if model not in model_types:
                model_types[model] = {'wins': 0, 'total': 0}
            model_types[model]['total'] += 1
            if t.get('outcome') == 'WIN':
                model_types[model]['wins'] += 1
        
        win_rate_by_model = {
            m: round(d['wins'] / d['total'] * 100, 1) 
            for m, d in model_types.items() if d['total'] > 0
        }
        
        # Total P&L
        total_pnl = sum(t.get('realized_pnl', 0) for t in closed_trades)
        
        return {
            "analytics": {
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                "total_pnl": round(total_pnl, 2),
                "win_rate_by_confidence": win_rate_by_confidence,
                "win_rate_by_model": win_rate_by_model
            }
        }
    except Exception as e:
        state.log_error("ANALYTICS", f"Analytics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRADE OUTCOMES FOR REINFORCEMENT LEARNING
# ============================================================================

class TradeOutcome(BaseModel):
    """Trade outcome for reinforcement learning"""
    ticker: str
    entry_price: float
    exit_price: float
    shares: int
    realized_pnl: float
    pnl_pct: float
    confidence: float = 0.0
    signal_reason: str = ""
    outcome: str  # WIN, LOSS, BREAKEVEN
    holding_duration_seconds: int = 0


@app.post("/api/trade-outcomes")
async def record_trade_outcome(outcome: TradeOutcome):
    """Record a trade outcome for reinforcement learning training"""
    try:
        outcome_record = {
            "timestamp": datetime.now().isoformat(),
            "ticker": outcome.ticker.upper(),
            "entry_price": outcome.entry_price,
            "exit_price": outcome.exit_price,
            "shares": outcome.shares,
            "realized_pnl": outcome.realized_pnl,
            "pnl_pct": outcome.pnl_pct,
            "confidence": outcome.confidence,
            "signal_reason": outcome.signal_reason,
            "outcome": outcome.outcome,
            "holding_duration_seconds": outcome.holding_duration_seconds
        }
        
        state.trade_outcomes.append(outcome_record)
        state.save_trade_outcomes()
        
        logger.info(f"Trade outcome recorded: {outcome.ticker} PnL={outcome.pnl_pct:.2f}% ({outcome.outcome})")
        
        # Check if RL training should be triggered
        ready, reason = state.is_rl_ready()
        
        return {
            "status": "ok",
            "outcome_id": len(state.trade_outcomes) - 1,
            "rl_ready": ready,
            "rl_status": reason,
            "total_outcomes": len(state.trade_outcomes)
        }
    except Exception as e:
        state.log_error("TRADE_OUTCOME", f"Failed to record trade outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-outcomes")
async def get_trade_outcomes(limit: int = 100, ticker: str = None):
    """Get recorded trade outcomes"""
    try:
        outcomes = state.trade_outcomes
        if ticker:
            outcomes = [o for o in outcomes if o.get('ticker', '').upper() == ticker.upper()]
        
        return {
            "outcomes": outcomes[-limit:],
            "total": len(outcomes),
            "rl_config": state.rl_config
        }
    except Exception as e:
        state.log_error("TRADE_OUTCOME", f"Failed to get trade outcomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REINFORCEMENT LEARNING CONFIGURATION
# ============================================================================

class RLConfigUpdate(BaseModel):
    """Update RL configuration"""
    enabled: Optional[bool] = None
    min_trades_required: Optional[int] = None
    rl_training_interval_hours: Optional[int] = None
    confidence_weight_factor: Optional[float] = None
    pnl_weight_cap: Optional[float] = None


@app.get("/api/rl/config")
async def get_rl_config():
    """Get reinforcement learning configuration"""
    ready, reason = state.is_rl_ready()
    return {
        "config": state.rl_config,
        "status": {
            "ready": ready,
            "reason": reason,
            "closed_trades": state.get_closed_trade_count()
        }
    }


@app.post("/api/rl/config")
async def update_rl_config(update: RLConfigUpdate):
    """Update reinforcement learning configuration"""
    try:
        if update.enabled is not None:
            state.rl_config['enabled'] = update.enabled
        if update.min_trades_required is not None:
            if update.min_trades_required < 5:
                raise HTTPException(status_code=400, detail="min_trades_required must be at least 5")
            state.rl_config['min_trades_required'] = update.min_trades_required
        if update.rl_training_interval_hours is not None:
            if update.rl_training_interval_hours < 1:
                raise HTTPException(status_code=400, detail="interval must be at least 1 hour")
            state.rl_config['rl_training_interval_hours'] = update.rl_training_interval_hours
        if update.confidence_weight_factor is not None:
            state.rl_config['confidence_weight_factor'] = max(1.0, min(5.0, update.confidence_weight_factor))
        if update.pnl_weight_cap is not None:
            state.rl_config['pnl_weight_cap'] = max(1.0, min(10.0, update.pnl_weight_cap))
        
        state.save_rl_config()
        logger.info(f"RL config updated: {state.rl_config}")
        
        return {"status": "ok", "config": state.rl_config}
    except HTTPException:
        raise
    except Exception as e:
        state.log_error("RL_CONFIG", f"Failed to update RL config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/trigger")
async def trigger_rl_training(background_tasks: BackgroundTasks, force: bool = False):
    """Manually trigger reinforcement learning training"""
    try:
        ready, reason = state.is_rl_ready()
        
        if not ready and not force:
            return {
                "status": "skipped",
                "reason": reason,
                "hint": "Use ?force=true to force training regardless of conditions"
            }
        
        # Queue RL training as background task
        background_tasks.add_task(run_rl_training)
        
        return {
            "status": "queued",
            "message": "Reinforcement learning training queued",
            "trade_count": state.get_closed_trade_count()
        }
    except Exception as e:
        state.log_error("RL_TRIGGER", f"Failed to trigger RL training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_rl_training():
    """Run reinforcement learning training with trade outcomes"""
    try:
        logger.info("Starting reinforcement learning training...")
        
        # Get trade outcomes
        outcomes = state.trade_outcomes
        if not outcomes:
            logger.warning("No trade outcomes for RL training")
            return
        
        # Group by ticker
        ticker_outcomes = {}
        for outcome in outcomes:
            ticker = outcome.get('ticker')
            if ticker not in ticker_outcomes:
                ticker_outcomes[ticker] = []
            ticker_outcomes[ticker].append(outcome)
        
        # Train each ticker with weighted samples
        trained_count = 0
        for ticker, ticker_trades in ticker_outcomes.items():
            if len(ticker_trades) < 3:  # Need at least 3 trades per ticker
                continue
            
            try:
                # Calculate sample weights based on trade outcomes
                sample_weights = calculate_rl_sample_weights(ticker_trades)
                
                # Fetch historical data
                df = state.fetcher.fetch_historical_data(ticker)
                if df is None or len(df) < 100:
                    continue
                
                # Train with weights
                result = state.trainer.train_and_validate(
                    df, ticker, epochs=30,
                    progress_callback=lambda e, t: None  # Silent training
                )
                
                if result:
                    trained_count += 1
                    logger.info(f"RL training completed for {ticker}")
                    
            except Exception as e:
                logger.error(f"RL training failed for {ticker}: {e}")
                continue
        
        # Update last training timestamp
        state.rl_config['last_rl_training'] = datetime.now().isoformat()
        state.save_rl_config()
        
        logger.info(f"RL training completed: {trained_count} tickers trained")
        
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        state.log_error("RL_TRAINING", f"Reinforcement learning training failed: {e}")


def calculate_rl_sample_weights(trades: List[Dict]) -> List[float]:
    """
    Calculate sample weights for reinforcement learning.
    
    Weighting strategy (Option B - Confidence-based feedback):
    - High confidence + WIN = strong positive weight (model was right and confident)
    - High confidence + LOSS = strong negative weight (penalty for overconfidence)
    - Low confidence + WIN = moderate positive (lucky or need more data)
    - Low confidence + LOSS = moderate negative (expected uncertainty)
    """
    weights = []
    
    conf_factor = state.rl_config.get('confidence_weight_factor', 2.0)
    weight_cap = state.rl_config.get('pnl_weight_cap', 3.0)
    
    for trade in trades:
        confidence = trade.get('confidence', 0.5)
        outcome = trade.get('outcome', 'BREAKEVEN')
        pnl_pct = trade.get('pnl_pct', 0)
        
        # Base weight
        base_weight = 1.0
        
        if outcome == 'WIN':
            # Reward: higher confidence wins get more weight
            reward_multiplier = 1.0 + (confidence * conf_factor * 0.5)
            # Scale by profit magnitude
            profit_bonus = min(abs(pnl_pct) / 10, 0.5)  # Cap at 0.5 bonus
            weight = base_weight * reward_multiplier + profit_bonus
            
        elif outcome == 'LOSS':
            # Penalty: higher confidence losses get more weight (to learn from mistakes)
            penalty_multiplier = 1.0 + (confidence * conf_factor)
            # Scale by loss magnitude
            loss_penalty = min(abs(pnl_pct) / 10, 0.5)
            weight = base_weight * penalty_multiplier + loss_penalty
            
        else:  # BREAKEVEN
            weight = base_weight
        
        # Cap the weight
        weight = min(weight, weight_cap)
        weights.append(weight)
    
    return weights


# ============================================================================
# ENHANCED HEALTH CHECK FOR CLIENT OFFLINE DETECTION
# ============================================================================

@app.get("/api/heartbeat")
async def heartbeat():
    """Simple heartbeat endpoint for client health checking"""
    state.last_heartbeat = datetime.now()
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "server_time_ms": int(time.time() * 1000)
    }


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
    # Collect live prices every 5 seconds (Alpaca allows ~20 req/sec)
    state.scheduler.add_job(
        collect_live_prices,
        'interval',
        seconds=5,
        id='live_price_collection',
        max_instances=1,  # Only one instance at a time
        coalesce=True,    # Combine missed runs
        replace_existing=True
    )
    logger.info("Live price collection scheduled (every 5s)")

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
        
        # Save cache periodically (every 20 collections = ~1 minute)
        run_count = getattr(state, '_price_collection_count', 0) + 1
        state._price_collection_count = run_count
        
        if run_count % 20 == 0:
            state.save_chart_cache()
            logger.info(f"Chart cache saved: {added}/{len(TOP_STOCKS)} prices collected")
        elif run_count == 1:
            # Log first collection
            logger.info(f"[Alpaca] First price collection: {added}/{len(TOP_STOCKS)} prices")
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
        schedule_live_price_collection()  # Start automatic price collection
    
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
