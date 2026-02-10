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
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import logging
import base64
import secrets
import hmac
import psutil

# Central timezone for all timestamps
TZ_BERLIN = ZoneInfo("Europe/Berlin")

def now() -> datetime:
    """Return current time in Europe/Berlin timezone."""
    return datetime.now(TZ_BERLIN)

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request, Response, Form, Path as FastapiPath, WebSocket, WebSocketDisconnect
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
from universal_trainer import UniversalModelTrainer, UNIVERSAL_MODEL_FILE, UNIVERSAL_SCALER_FILE
from risk_profiles import (
    RiskProfile, RiskManager, PROFILES, CONSERVATIVE_PROFILE,
    BALANCED_PROFILE, AGGRESSIVE_PROFILE, interpolate_profile
)
from simulator import TradingSimulator, create_simulator
from ensemble import EnsemblePredictor, ModelPrediction, EnsemblePrediction
from model_registry import ModelRegistry
from rate_limiter import RateLimiter
from metrics_collector import metrics as prom_metrics
from ab_testing import ABTestingManager

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
# NOTE: allow_credentials=True is incompatible with allow_origins=["*"]
# in newer Starlette versions and causes WebSocket 403 errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
# REQUEST METRICS MIDDLEWARE (pure ASGI — avoids BaseHTTPMiddleware h11 bug)
# ============================================================================

class MetricsASGIMiddleware:
    """Pure ASGI middleware for request timing headers and Prometheus metrics.

    Using a raw ASGI middleware instead of @app.middleware('http') avoids the
    ``h11.LocalProtocolError: Can't send data when our state is ERROR`` that
    occurs when BaseHTTPMiddleware tries to stream a response on a connection
    that has already errored (e.g. client disconnect).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 500  # default until we see the real status

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
                process_time = time.time() - start_time
                # Inject timing headers into the response
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", f"{process_time:.3f}s".encode()))
                headers.append((b"x-server-time", now().isoformat().encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            raise
        finally:
            # Record Prometheus metrics regardless of success/failure
            process_time = time.time() - start_time
            try:
                path = scope.get("path", "/unknown")
                method = scope.get("method", "GET")
                # Normalise parameterised paths for metric cardinality
                if path.startswith("/api/models/") and "/download" in path:
                    path = "/api/models/{ticker}/download"
                elif path.startswith("/api/models/") and "/hash" in path:
                    path = "/api/models/{ticker}/hash"
                elif path.startswith("/api/chart-cache/"):
                    path = "/api/chart-cache/{ticker}"
                prom_metrics.http_requests_total.inc(
                    method=method, path=path, status=str(status_code),
                )
                prom_metrics.http_request_duration.observe(
                    process_time, method=method, path=path,
                )
            except Exception:
                pass


app.add_middleware(MetricsASGIMiddleware)


# Login auth for web UI
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change_this_secret")
SESSION_COOKIE = "ai_trading_session"

# API Key authentication for /api/* endpoints
API_KEY = os.getenv("API_KEY", "changeme_api_key")
# Endpoints that do NOT require an API key
API_PUBLIC_ENDPOINTS = {"/api/health", "/api/heartbeat"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Require X-API-Key header for all /api/* endpoints (except public ones).
    Logged-in dashboard users (valid session cookie) bypass the API key check."""
    path = request.url.path
    # Strip trailing slashes for consistent matching
    clean_path = path.rstrip("/")
    if clean_path.startswith("/api") and clean_path not in API_PUBLIC_ENDPOINTS:
        # Allow logged-in UI users to call API without key
        if _is_logged_in(request):
            return await call_next(request)
        provided_key = request.headers.get("X-API-Key", "")
        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing API key. Set X-API-Key header."},
            )
        try:
            if not hmac.compare_digest(provided_key, API_KEY):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid API key."},
                )
        except (TypeError, ValueError) as e:
            logger.warning(f"API key comparison error: {e}")
            return JSONResponse(
                status_code=401,
                content={"error": "API key validation failed."},
            )
    return await call_next(request)


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

# ── Atomic JSON I/O (crash-safe) ──────────────────────────────────────────────
def _atomic_json_write(filepath: Path, data, **kwargs) -> None:
    """Write JSON atomically: write to temp file, then rename (POSIX-safe)."""
    tmp = filepath.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, **kwargs)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(filepath)  # atomic on POSIX

def _safe_json_load(filepath: Path, default=None):
    """Load JSON, returning *default* on missing / empty / corrupt file."""
    if default is None:
        default = {}
    try:
        if not filepath.exists() or filepath.stat().st_size == 0:
            return default
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load %s: %s", filepath, e)
        return default


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
        
        # Universal model state (must be set before _load_model_versions)
        self.universal_model_version = "v0"
        self.universal_training_status = "idle"  # idle/training/completed/failed
        self.universal_training_progress = 0.0
        self.universal_training_message = ""
        self.universal_training_start = None
        self.universal_training_metrics = {}
        
        # Model version tracking
        self.model_versions_file = Path("./data/model_versions.json")
        self.model_versions_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_universal_version()  # must run before _load_model_versions
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
        self.universal_trainer = UniversalModelTrainer(lookback=20, epochs=150)
        
        # Track startup time for uptime calculation
        self.startup_time = now()
        
        # Heartbeat tracking for client health detection
        self.last_heartbeat = now()
    
    def _load_feature_stats(self):
        """Load feature statistics from disk"""
        self.feature_stats = _safe_json_load(self.feature_stats_file, {})
        if self.feature_stats:
            logger.info(f"Loaded feature stats for {len(self.feature_stats)} tickers")
    
    def save_feature_stats(self):
        """Save feature statistics to disk (atomic write)"""
        try:
            _atomic_json_write(self.feature_stats_file, self.feature_stats, indent=2)
        except Exception as e:
            logger.error(f"Could not save feature stats: {e}")
    
    def _load_trade_journal(self):
        """Load trade journal from disk"""
        self.trade_journal = _safe_json_load(self.trade_journal_file, [])
        if self.trade_journal:
            logger.info(f"Loaded {len(self.trade_journal)} trade records")
    
    def save_trade_journal(self):
        """Save trade journal to disk (atomic write)"""
        try:
            _atomic_json_write(self.trade_journal_file, self.trade_journal, indent=2)
        except Exception as e:
            logger.error(f"Could not save trade journal: {e}")
    
    def _load_trade_outcomes(self):
        """Load trade outcomes for reinforcement learning"""
        self.trade_outcomes = _safe_json_load(self.trade_outcomes_file, [])
        if self.trade_outcomes:
            logger.info(f"Loaded {len(self.trade_outcomes)} trade outcomes for RL")
    
    def save_trade_outcomes(self):
        """Save trade outcomes to disk (atomic write)"""
        try:
            _atomic_json_write(self.trade_outcomes_file, self.trade_outcomes, indent=2)
        except Exception as e:
            logger.error(f"Could not save trade outcomes: {e}")
    
    def _load_rl_config(self):
        """Load reinforcement learning configuration"""
        saved_config = _safe_json_load(self.rl_config_file, {})
        if saved_config:
            self.rl_config.update(saved_config)
            logger.info(f"Loaded RL config: enabled={self.rl_config['enabled']}, min_trades={self.rl_config['min_trades_required']}")
    
    def save_rl_config(self):
        """Save reinforcement learning configuration (atomic write)"""
        try:
            _atomic_json_write(self.rl_config_file, self.rl_config, indent=2)
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
                hours_since = (now() - last_dt).total_seconds() / 3600
                interval = self.rl_config.get('rl_training_interval_hours', 24)
                if hours_since < interval:
                    return False, f"Training interval not reached: {hours_since:.1f}/{interval}h"
            except Exception:
                pass
        
        return True, f"Ready with {closed_count} closed trades"

    def log_error(self, error_type: str, message: str, details: dict = None):
        """Log an error for diagnostics"""
        error_entry = {
            "timestamp": now().isoformat(),
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
        data = _safe_json_load(self.model_versions_file, {})
        self.active_models = data.get('versions', {})

        # Backfill for existing model files (any variant: AAPL_model, AAPL_deep_model, ...)
        try:
            for model_file in self.models_dir.glob("*_model.h5"):
                ticker = model_file.stem.replace("_model", "")
                # Strip ensemble suffixes to get base ticker
                for suffix in ('_deep', '_fast', '_wide'):
                    if ticker.endswith(suffix):
                        ticker = ticker[:-len(suffix)]
                        break
                if ticker not in self.active_models:
                    self.active_models[ticker] = "v1"
        except Exception as e:
            logger.warning(f"Could not backfill model versions: {e}")

        self.save_model_versions()

    def save_model_versions(self):
        """Save model versions to disk (atomic write)"""
        try:
            _atomic_json_write(self.model_versions_file, {
                'versions': self.active_models,
                'universal_version': self.universal_model_version,
                'last_updated': now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Could not save model versions: {e}")
    
    def _load_universal_version(self):
        """Load universal model version from disk"""
        try:
            if self.model_versions_file.exists():
                with open(self.model_versions_file, 'r') as f:
                    data = json.load(f)
                    self.universal_model_version = data.get('universal_version', 'v0')
            # Check if model file exists
            if Path(UNIVERSAL_MODEL_FILE).exists() and self.universal_model_version == 'v0':
                self.universal_model_version = 'v1'
        except Exception:
            pass
    
    def _load_chart_cache(self):
        """Load cached chart data from disk"""
        data = _safe_json_load(self.chart_cache_file, {})
        self.live_prices_cache = data.get('live_prices', {})
        if self.live_prices_cache:
            logger.info(f"Loaded chart cache with {len(self.live_prices_cache)} tickers")
    
    def save_chart_cache(self):
        """Save chart data cache to disk (atomic write)"""
        try:
            _atomic_json_write(self.chart_cache_file, {
                'live_prices': self.live_prices_cache,
                'last_updated': now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Could not save chart cache: {e}")
    
    def add_live_price(self, ticker: str, price: float):
        """Add a live price point to the cache"""
        if ticker not in self.live_prices_cache:
            self.live_prices_cache[ticker] = []
        
        self.live_prices_cache[ticker].append({
            'price': price,
            'timestamp': now().isoformat()
        })
        
        # Limit cache size
        if len(self.live_prices_cache[ticker]) > self.live_prices_max_points:
            self.live_prices_cache[ticker] = self.live_prices_cache[ticker][-self.live_prices_max_points:]

state = ServerState()

# Initialize risk manager and simulator after state
risk_manager = RiskManager()
simulator = create_simulator(state.fetcher)

# S1: Ensemble predictor
ensemble_predictor = EnsemblePredictor(strategy="weighted_average")

# S2: Model registry (versioning & rollback)
model_registry = ModelRegistry(
    models_dir="./models",
    registry_file="./data/model_registry.json",
    max_versions_per_ticker=10,
)

# S3: WebSocket rate limiter
ws_rate_limiter = RateLimiter(
    rate=5.0,
    burst=20,
    max_connections_per_ip=5,
    ban_threshold=50,
    ban_duration_s=300.0,
)

# S5: A/B testing manager
ab_manager = ABTestingManager(experiments_file="./data/ab_experiments.json")


# ============================================================================
# WEBSOCKET SUPPORT
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)


ws_manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time client updates"""
    # S3: Rate-limit connection
    client_ip = websocket.client.host if websocket.client else "unknown"
    ws_id = f"{client_ip}:{id(websocket)}"
    allowed, reason = ws_rate_limiter.check_connection(client_ip, ws_id)
    if not allowed:
        await websocket.close(code=1008, reason=reason)
        logger.warning("WebSocket connection rejected for %s: %s", client_ip, reason)
        return

    await ws_manager.connect(websocket)
    ws_rate_limiter.register_connection(client_ip, ws_id)
    prom_metrics.ws_connections.inc()
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to AI Trading Server",
            "timestamp": now().isoformat()
        })
        
        # Send current prices if available
        if state.live_prices_cache:
            prices = {}
            for ticker, points in state.live_prices_cache.items():
                if points:
                    prices[ticker] = points[-1].get('price', 0)
            if prices:
                await websocket.send_json({
                    "type": "prices",
                    "prices": prices,
                    "timestamp": now().isoformat()
                })
        
        # Send current training status
        if state.training_status:
            await websocket.send_json({
                "type": "training_status",
                "data": state.training_status,
                "timestamp": now().isoformat()
            })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                # S3: Rate-limit messages
                msg_ok, msg_reason = ws_rate_limiter.allow_message(client_ip)
                if not msg_ok:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Rate limited: {msg_reason}",
                    })
                    continue

                prom_metrics.ws_messages_total.inc(direction="inbound")
                action = data.get('action') or data.get('type', '')
                
                if action == 'ping':
                    await websocket.send_json({"type": "pong"})
                
                elif action == 'subscribe':
                    # Acknowledge subscription
                    channel = data.get('channel', 'all')
                    await websocket.send_json({
                        "type": "subscribed",
                        "channel": channel
                    })
                
                elif action == 'get_prices':
                    prices = {}
                    for ticker, points in state.live_prices_cache.items():
                        if points:
                            prices[ticker] = points[-1].get('price', 0)
                    await websocket.send_json({
                        "type": "prices",
                        "prices": prices,
                        "timestamp": now().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)
        ws_rate_limiter.unregister_connection(client_ip, ws_id)
        prom_metrics.ws_connections.dec()


async def broadcast_training_update(ticker: str, status: str, progress: float = 0, metrics: dict = None):
    """Broadcast training update to all WebSocket clients"""
    message = {
        "type": "training_update",
        "ticker": ticker,
        "status": status,
        "progress": progress,
        "timestamp": now().isoformat()
    }
    if metrics:
        message["metrics"] = metrics
    await ws_manager.broadcast(message)


async def broadcast_prices(prices: dict):
    """Broadcast live prices to all WebSocket clients"""
    await ws_manager.broadcast({
        "type": "prices",
        "prices": prices,
        "timestamp": now().isoformat()
    })


def _ws_broadcast(msg_type: str, **kwargs):
    """Helper to broadcast WebSocket messages from sync code (background tasks).
    Safely schedules the coroutine on the running event loop."""
    if not ws_manager.active_connections:
        return
    message = {"type": msg_type, "timestamp": now().isoformat()}
    message.update(kwargs)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(message), loop)
        else:
            loop.run_until_complete(ws_manager.broadcast(message))
    except RuntimeError:
        # No event loop available (shouldn't happen with uvicorn)
        pass


class InMemoryLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        state.logs.append({"time": now().isoformat(), "message": msg})
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


# Risk Management Models
class RiskProfileUpdate(BaseModel):
    profile_name: Optional[str] = None  # 'conservative', 'balanced', 'aggressive'
    risk_level: Optional[int] = Field(None, ge=1, le=10)
    overrides: Optional[Dict[str, Any]] = None


class SimulationRequest(BaseModel):
    tickers: Optional[List[str]] = None  # Defaults to TOP_STOCKS
    profile_name: Optional[str] = "balanced"
    risk_level: Optional[int] = Field(None, ge=1, le=10)
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    initial_capital: Optional[float] = 100000.0
    use_predictions: Optional[bool] = False


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Server health check"""
    uptime = (now() - state.startup_time).total_seconds()
    return {
        "status": "healthy",
        "timestamp": now().isoformat(),
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
        "last_updated": now().isoformat(),
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
            "timestamp": now().isoformat(),
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
                trade['closed_at'] = now().isoformat()
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
            "timestamp": now().isoformat(),
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
        state.rl_config['last_rl_training'] = now().isoformat()
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
# RISK MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/risk/profiles")
async def get_risk_profiles():
    """Get all available risk profiles"""
    profiles_info = {}
    for name, profile in PROFILES.items():
        profiles_info[name] = {
            "name": profile.name,
            "level": profile.level,
            "description": profile.description,
            "position_size_default": profile.position_size_default,
            "stop_loss_default": profile.stop_loss_default,
            "take_profit_default": profile.take_profit_default,
            "risk_reward_ratio": profile.risk_reward_ratio,
            "max_concurrent_trades": profile.max_concurrent_trades,
            "min_confidence": profile.min_confidence
        }
    return {"profiles": profiles_info}


@app.get("/api/risk/current")
async def get_current_risk_profile():
    """Get current risk profile settings"""
    profile = risk_manager.get_active_profile()
    return {
        "profile": {
            "name": profile.name,
            "level": profile.level,
            "description": profile.description,
            "position_size_min": profile.position_size_min,
            "position_size_max": profile.position_size_max,
            "position_size_default": profile.position_size_default,
            "stop_loss_tight": profile.stop_loss_tight,
            "stop_loss_wide": profile.stop_loss_wide,
            "stop_loss_default": profile.stop_loss_default,
            "take_profit_min": profile.take_profit_min,
            "take_profit_max": profile.take_profit_max,
            "take_profit_default": profile.take_profit_default,
            "risk_reward_ratio": profile.risk_reward_ratio,
            "max_concurrent_trades": profile.max_concurrent_trades,
            "min_confidence": profile.min_confidence,
            "long_entry_threshold": profile.long_entry_threshold,
            "short_entry_threshold": profile.short_entry_threshold,
        },
        "overrides": risk_manager.custom_overrides,
        "effective_settings": {
            "position_size": getattr(risk_manager.get_active_profile(), 'position_size_default', 0),
            "stop_loss": getattr(risk_manager.get_active_profile(), 'stop_loss_default', 0),
            "take_profit": getattr(risk_manager.get_active_profile(), 'take_profit_default', 0),
            "max_concurrent_trades": getattr(risk_manager.get_active_profile(), 'max_concurrent_trades', 0),
            "min_confidence": getattr(risk_manager.get_active_profile(), 'min_confidence', 0)
        }
    }


@app.post("/api/risk/profile")
async def update_risk_profile(update: RiskProfileUpdate):
    """Update risk profile settings"""
    try:
        if update.profile_name:
            success = risk_manager.set_profile(update.profile_name)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid profile name: {update.profile_name}. Valid options: {list(PROFILES.keys())}"
                )
            logger.info(f"Risk profile changed to: {update.profile_name}")
        
        if update.risk_level is not None:
            risk_manager.set_custom_level(update.risk_level)
            logger.info(f"Risk level set to: {update.risk_level}")
        
        if update.overrides:
            for key, value in update.overrides.items():
                if value is None:
                    # Clear single override by re-setting all others
                    if key in risk_manager.custom_overrides:
                        del risk_manager.custom_overrides[key]
                        risk_manager.save_config()
                else:
                    risk_manager.set_override(key, value)
            logger.info(f"Risk overrides updated: {update.overrides}")
        
        return await get_current_risk_profile()
    except HTTPException:
        raise
    except Exception as e:
        state.log_error("RISK_PROFILE", f"Failed to update risk profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/risk/overrides")
async def clear_risk_overrides():
    """Clear all risk parameter overrides"""
    risk_manager.clear_overrides()
    logger.info("All risk overrides cleared")
    return {"status": "ok", "message": "All overrides cleared"}


@app.get("/api/risk/calculate")
async def calculate_risk_params(
    capital: float = 100000,
    confidence: float = 0.7,
    current_price: float = 100.0,
    expected_change: float = 2.0,
    is_long: bool = True
):
    """Calculate risk parameters for a potential trade"""
    position_size = risk_manager.calculate_position_size(capital, confidence)
    stop_loss = risk_manager.get_stop_loss(current_price, is_long=is_long)
    take_profit = risk_manager.get_take_profit(current_price, is_long=is_long)
    should_enter, reason = risk_manager.should_enter_trade(
        expected_change=expected_change,
        confidence=confidence,
        current_open_trades=0,
        is_long=is_long
    )
    
    return {
        "position_size": position_size,
        "position_size_pct": position_size / capital * 100,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "should_enter": should_enter,
        "entry_reason": reason,
        "risk_per_share": abs(current_price - stop_loss),
        "reward_per_share": abs(take_profit - current_price)
    }


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@app.get("/api/simulation/status")
async def get_simulation_status():
    """Get current simulation status"""
    return simulator.get_status()


@app.get("/api/simulation/history")
async def get_simulation_history(limit: int = 10):
    """Get simulation history"""
    return {"history": simulator.get_history(limit)}


@app.post("/api/simulation/run")
async def run_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """Run a trading simulation"""
    try:
        if simulator.is_running:
            raise HTTPException(status_code=409, detail="Simulation already running")
        
        # Determine profile
        if request.risk_level is not None:
            profile = interpolate_profile(request.risk_level)
        elif request.profile_name:
            profile = PROFILES.get(request.profile_name.lower())
            if not profile:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid profile: {request.profile_name}. Valid: {list(PROFILES.keys())}"
                )
        else:
            profile = risk_manager.get_active_profile()
        
        # Default tickers
        tickers = request.tickers or TOP_STOCKS[:20]
        
        def run_sim():
            try:
                result = simulator.run_simulation(
                    tickers=tickers,
                    profile=profile,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    use_predictions=request.use_predictions
                )
                logger.info(f"Simulation complete: {result.total_trades} trades, "
                           f"{result.total_return_pct:.2f}% return")
            except Exception as e:
                logger.error(f"Simulation failed: {e}")
        
        background_tasks.add_task(run_sim)
        
        return {
            "status": "started",
            "profile": profile.name,
            "risk_level": profile.level,
            "tickers": len(tickers),
            "initial_capital": request.initial_capital
        }
    except HTTPException:
        raise
    except Exception as e:
        state.log_error("SIMULATION", f"Failed to start simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulation/compare")
async def compare_profiles(background_tasks: BackgroundTasks):
    """Compare all risk profiles via simulation"""
    try:
        if simulator.is_running:
            raise HTTPException(status_code=409, detail="Simulation already running")
        
        tickers = TOP_STOCKS[:15]
        
        def run_compare():
            try:
                result = simulator.compare_profiles(tickers)
                logger.info(f"Profile comparison complete. Best: {result.get('best_profile')}")
            except Exception as e:
                logger.error(f"Profile comparison failed: {e}")
        
        background_tasks.add_task(run_compare)
        
        return {"status": "started", "message": "Comparing all profiles..."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulation/optimize")
async def optimize_risk_level(background_tasks: BackgroundTasks):
    """Find optimal risk level via simulation"""
    try:
        if simulator.is_running:
            raise HTTPException(status_code=409, detail="Simulation already running")
        
        tickers = TOP_STOCKS[:15]
        
        def run_optimize():
            try:
                result = simulator.optimize_level(tickers)
                logger.info(f"Optimization complete. Optimal level: {result.get('optimal_level')}")
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
        
        background_tasks.add_task(run_optimize)
        
        return {"status": "started", "message": "Optimizing risk level..."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENHANCED HEALTH CHECK FOR CLIENT OFFLINE DETECTION
# ============================================================================

@app.get("/api/heartbeat")
async def heartbeat():
    """Simple heartbeat endpoint for client health checking"""
    state.last_heartbeat = now()
    return {
        "status": "alive",
        "timestamp": now().isoformat(),
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
    """List all available models (per-ticker + universal)"""
    models = []
    
    # Universal model
    universal_path = Path(UNIVERSAL_MODEL_FILE)
    if universal_path.exists():
        file_stat = universal_path.stat()
        with open(universal_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        models.append({
            "ticker": "UNIVERSAL",
            "version": state.universal_model_version,
            "trained_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "file_hash": file_hash,
            "file_size": file_stat.st_size
        })
    
    # Per-ticker models
    for model_file in state.models_dir.glob("*_model.h5"):
        ticker = model_file.stem.replace("_model", "")
        if ticker == "universal_market":
            continue  # Already listed above
        file_stat = model_file.stat()
        
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
        state.training_start[ticker] = now()
        state.training_progress[ticker] = 0.0
        
        # Broadcast training started via WebSocket
        _ws_broadcast("training_update", ticker=ticker, status="started")
        
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
            _ws_broadcast("training_update", ticker=ticker, status="progress", progress=state.training_progress[ticker])

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
            state.last_training[ticker] = now().isoformat()
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
            duration = (now() - state.training_start[ticker]).total_seconds()
            state.training_history.append({
                "ticker": ticker,
                "status": "completed",
                "duration_seconds": duration,
                "trained_at": state.last_training[ticker],
                "class_accuracy": result.get("class_accuracy"),
                "reg_mae": result.get("reg_mae")
            })
            logger.info(f"✅ Completed training for {ticker} in {duration:.0f}s")
            
            # S2: Register in model registry
            try:
                model_registry.register(
                    ticker=ticker,
                    metrics=state.training_metrics.get(ticker, {}),
                    training_duration_s=duration,
                )
            except Exception as reg_err:
                logger.warning(f"Model registry failed for {ticker}: {reg_err}")
            
            # S4: Prometheus training metrics
            prom_metrics.trainings_total.inc(ticker=ticker, status="completed")
            prom_metrics.training_duration.observe(duration, ticker=ticker, model_type="per_ticker")
            
            # Broadcast completion via WebSocket
            _ws_broadcast("training_update", ticker=ticker, status="completed", 
                         metrics=state.training_metrics.get(ticker, {}))
        else:
            state.training_status[ticker] = "failed"
            state.training_progress[ticker] = 0.0
            duration = (now() - state.training_start[ticker]).total_seconds()
            state.training_history.append({
                "ticker": ticker,
                "status": "failed",
                "duration_seconds": duration,
                "trained_at": now().isoformat()
            })
            logger.error(f"Training failed for {ticker}")
            prom_metrics.trainings_total.inc(ticker=ticker, status="failed")
            _ws_broadcast("training_update", ticker=ticker, status="failed")
            
    except Exception as e:
        import traceback
        state.training_status[ticker] = "failed"
        duration = (now() - state.training_start[ticker]).total_seconds() if ticker in state.training_start else None
        state.training_history.append({
            "ticker": ticker,
            "status": "failed",
            "duration_seconds": duration,
            "trained_at": now().isoformat(),
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
    """Start training for multiple stocks (legacy per-ticker)"""
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
    """Train the universal model (one model for all stocks)"""
    if state.universal_training_status == "training":
        raise HTTPException(status_code=409, detail="Universal training already in progress")
    
    state.universal_training_status = "training"
    state.universal_training_progress = 0.0
    state.universal_training_message = "Queued..."
    background_tasks.add_task(train_universal_task)
    return {"status": "universal training started", "tickers": len(TOP_STOCKS)}


def train_universal_task():
    """Background task: train one universal model across all tickers"""
    try:
        state.universal_training_start = now()
        state.universal_training_status = "training"
        logger.info(f"[Universal] Starting training with {len(TOP_STOCKS)} tickers...")
        _ws_broadcast("training_update", status="started", mode="universal",
                       message=f"Training universal model with {len(TOP_STOCKS)} tickers")

        def on_progress(phase, pct, msg):
            state.universal_training_progress = pct
            state.universal_training_message = msg
            _ws_broadcast("training_update", status="progress", mode="universal",
                           progress=pct, phase=phase, message=msg)

        result = state.universal_trainer.train(
            fetcher=state.fetcher,
            tickers=TOP_STOCKS,
            progress_callback=on_progress,
        )

        if result:
            # Increment version
            try:
                cur = int(str(state.universal_model_version).lstrip("v"))
            except ValueError:
                cur = 0
            state.universal_model_version = f"v{cur + 1}"

            state.universal_training_status = "completed"
            state.universal_training_progress = 100.0
            state.universal_training_message = "Training completed"
            state.universal_training_metrics = result
            state.save_model_versions()

            duration = (now() - state.universal_training_start).total_seconds()
            state.training_history.append({
                "ticker": "UNIVERSAL",
                "status": "completed",
                "duration_seconds": duration,
                "trained_at": now().isoformat(),
                "class_accuracy": result.get("class_accuracy"),
                "reg_mae": result.get("reg_mae"),
                "tickers_used": result.get("tickers_used"),
                "total_samples": result.get("total_samples"),
            })
            logger.info(f"[Universal] ✅ Training completed in {duration:.0f}s "
                         f"(acc={result['class_accuracy']:.2%})")
            _ws_broadcast("training_update", status="completed", mode="universal",
                           metrics=result)
        else:
            state.universal_training_status = "failed"
            state.universal_training_progress = 0.0
            state.universal_training_message = "Training failed"
            logger.error("[Universal] Training failed")
            _ws_broadcast("training_update", status="failed", mode="universal")

    except Exception as e:
        import traceback
        state.universal_training_status = "failed"
        state.universal_training_message = f"Error: {str(e)}"
        logger.error(f"[Universal] Training error: {e}")
        logger.error(traceback.format_exc())
        _ws_broadcast("training_update", status="failed", mode="universal",
                       message=str(e))


@app.get("/api/training-status/universal")
async def universal_training_status():
    """Get universal model training status"""
    return {
        "status": state.universal_training_status,
        "progress": state.universal_training_progress,
        "message": state.universal_training_message,
        "version": state.universal_model_version,
        "metrics": state.universal_training_metrics,
        "model_exists": Path(UNIVERSAL_MODEL_FILE).exists(),
    }


@app.get("/api/models/universal/download")
async def download_universal_model():
    """Download the universal model + scaler as ZIP"""
    import shutil

    model_path = Path(UNIVERSAL_MODEL_FILE)
    scaler_path = Path(UNIVERSAL_SCALER_FILE)

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Universal model not found")

    temp_dir = Path("models/_universal_temp")
    temp_dir.mkdir(exist_ok=True)
    shutil.copy(model_path, temp_dir / "universal_market_model.h5")
    if scaler_path.exists():
        shutil.copy(scaler_path, temp_dir / "universal_scaler.pkl")

    zip_path = Path("models/universal_model")
    shutil.make_archive(str(zip_path), 'zip', str(temp_dir))
    shutil.rmtree(temp_dir)

    return FileResponse(
        path=str(zip_path) + ".zip",
        filename="universal_model.zip",
        media_type="application/zip",
    )


@app.get("/api/models/universal/hash")
async def universal_model_hash():
    """Get hash of universal model for cache invalidation"""
    model_path = Path(UNIVERSAL_MODEL_FILE)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Universal model not found")

    with open(model_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return {
        "hash": file_hash,
        "version": state.universal_model_version,
        "file_size": model_path.stat().st_size,
    }


@app.post("/api/train-all-legacy")
async def train_all_legacy(background_tasks: BackgroundTasks):
    """Legacy: Queue per-ticker training for all stocks"""
    for ticker in TOP_STOCKS:
        if state.training_status.get(ticker) not in ("training", "queued"):
            state.training_status[ticker] = "queued"
            if ticker not in state.training_queue:
                state.training_queue.append(ticker)
    background_tasks.add_task(process_training_queue)
    return {"status": "all queued (legacy per-ticker)", "count": len(TOP_STOCKS)}


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
# S4: PROMETHEUS METRICS ENDPOINT
# ============================================================================

@app.get("/metrics")
async def prometheus_metrics():
    """Expose metrics in Prometheus text exposition format."""
    try:
        uptime = (now() - state.startup_time).total_seconds()
        prom_metrics.update_system_metrics(uptime)
        prom_metrics.active_models.set(len(state.active_models))
        prom_metrics.training_queue_depth.set(len(state.training_queue))
    except Exception as e:
        logger.warning(f"Metrics update error: {e}")

    return Response(content=prom_metrics.render(), media_type="text/plain; version=0.0.4")


# ============================================================================
# S1: ENSEMBLE PREDICTION ENDPOINTS
# ============================================================================

class EnsembleConfigUpdate(BaseModel):
    strategy: Optional[str] = None


@app.get("/api/ensemble/config")
async def get_ensemble_config():
    """Get ensemble predictor configuration."""
    return {
        "strategy": ensemble_predictor.strategy,
        "available_strategies": list(EnsemblePredictor.STRATEGIES),
        "model_weights": ensemble_predictor.get_model_weights(),
    }


@app.post("/api/ensemble/config")
async def update_ensemble_config(update: EnsembleConfigUpdate):
    """Update ensemble predictor strategy."""
    try:
        if update.strategy:
            ensemble_predictor.set_strategy(update.strategy)
        return {"status": "ok", "strategy": ensemble_predictor.strategy}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ensemble/predict/{ticker}")
async def ensemble_predict(ticker: str):
    """Run ensemble prediction for a ticker using all available models.

    Collects predictions from per-ticker and universal models, then
    aggregates them via the configured ensemble strategy.
    """
    ticker = ticker.upper()

    predictions: List[ModelPrediction] = []
    start_t = time.time()

    # Per-ticker model prediction
    model_path = state.models_dir / f"{ticker}_model.h5"
    if model_path.exists():
        try:
            pred_start = time.time()
            df = state.fetcher.fetch_historical_data(ticker)
            if df is not None and len(df) >= 100:
                result = state.trainer.train_and_validate(df, ticker, epochs=0)
                if result:
                    signal = "UP" if result.get("predicted_change", 0) > 0.5 else (
                        "DOWN" if result.get("predicted_change", 0) < -0.5 else "NEUTRAL")
                    predictions.append(ModelPrediction(
                        model_id=f"{ticker}_per_ticker",
                        model_type="per_ticker",
                        signal=signal,
                        confidence=result.get("class_accuracy", 0.5),
                        predicted_change=result.get("predicted_change", 0),
                        version=state.active_models.get(ticker, "v1"),
                        latency_ms=(time.time() - pred_start) * 1000,
                    ))
        except Exception as e:
            logger.warning(f"Per-ticker prediction failed for {ticker}: {e}")

    # Universal model prediction (stub — real implementation would load and predict)
    universal_path = Path(UNIVERSAL_MODEL_FILE)
    if universal_path.exists():
        try:
            predictions.append(ModelPrediction(
                model_id="universal",
                model_type="universal",
                signal="NEUTRAL",
                confidence=0.5,
                predicted_change=0.0,
                version=state.universal_model_version,
                latency_ms=0.0,
            ))
        except Exception as e:
            logger.warning(f"Universal prediction failed for {ticker}: {e}")

    # Run ensemble
    result = ensemble_predictor.predict(ticker, predictions)
    total_ms = (time.time() - start_t) * 1000

    # S4: Track prediction metrics
    prom_metrics.predictions_total.inc(ticker=ticker, signal=result.signal)
    prom_metrics.prediction_latency.observe(total_ms / 1000, model_type="ensemble")
    prom_metrics.ensemble_predictions_total.inc(strategy=ensemble_predictor.strategy)

    return {
        "ticker": result.ticker,
        "signal": result.signal,
        "confidence": result.confidence,
        "predicted_change": result.predicted_change,
        "agreement_ratio": result.agreement_ratio,
        "strategy": result.strategy,
        "model_count": result.model_count,
        "individual_predictions": result.individual_predictions,
        "latency_ms": round(total_ms, 2),
    }


@app.post("/api/ensemble/feedback")
async def ensemble_feedback(
    ticker: str,
    actual_change: float,
):
    """Record actual market outcome for ensemble weight updates."""
    try:
        ensemble_predictor.record_outcome(ticker.upper(), actual_change, [])
        return {"status": "ok", "message": "Weights updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# S2: MODEL REGISTRY ENDPOINTS
# ============================================================================

@app.get("/api/registry/summary")
async def registry_summary():
    """Get model registry summary."""
    return model_registry.get_summary()


@app.get("/api/registry/{ticker}")
async def registry_versions(ticker: str):
    """List all versions for a ticker."""
    ticker = ticker.upper()
    versions = model_registry.get_all_versions(ticker)
    return {
        "ticker": ticker,
        "versions": [v.to_dict() for v in versions],
        "active": model_registry.get_active_version(ticker),
        "count": len(versions),
    }


@app.post("/api/registry/{ticker}/rollback")
async def registry_rollback(ticker: str, version: Optional[str] = None):
    """Rollback a ticker to a specific version (or previous)."""
    try:
        mv = model_registry.rollback(ticker.upper(), version)
        return {
            "status": "rolled_back",
            "ticker": mv.ticker,
            "version": mv.version,
            "file_hash": mv.file_hash,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# S3: RATE LIMITER ENDPOINTS
# ============================================================================

@app.get("/api/rate-limit/stats")
async def rate_limit_stats():
    """Get WebSocket rate limiter statistics."""
    return ws_rate_limiter.get_stats()


@app.post("/api/rate-limit/reset/{client_id}")
async def rate_limit_reset(client_id: str):
    """Reset rate-limit state for a client (admin action)."""
    ws_rate_limiter.reset_client(client_id)
    return {"status": "ok", "client_id": client_id}


# ============================================================================
# S5: A/B TESTING ENDPOINTS
# ============================================================================

class ABExperimentCreate(BaseModel):
    name: str
    control_version: str
    treatment_version: str
    ticker: str = ""
    description: str = ""
    control_traffic_pct: float = 50.0
    model_type: str = "per_ticker"


@app.get("/api/ab/experiments")
async def list_ab_experiments(status: Optional[str] = None):
    """List all A/B experiments."""
    return {"experiments": ab_manager.list_experiments(status)}


@app.post("/api/ab/experiments")
async def create_ab_experiment(data: ABExperimentCreate):
    """Create a new A/B experiment."""
    try:
        exp = ab_manager.create_experiment(
            name=data.name,
            control_version=data.control_version,
            treatment_version=data.treatment_version,
            ticker=data.ticker,
            description=data.description,
            control_traffic_pct=data.control_traffic_pct,
            model_type=data.model_type,
        )
        prom_metrics.ab_assignments_total.inc(experiment=exp.experiment_id, group="created")
        return {"status": "created", "experiment": exp.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ab/experiments/{experiment_id}")
async def get_ab_results(experiment_id: str):
    """Get results for an A/B experiment."""
    results = ab_manager.get_results(experiment_id)
    if not results:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return results


@app.post("/api/ab/experiments/{experiment_id}/complete")
async def complete_ab_experiment(experiment_id: str):
    """Complete an A/B experiment and return results."""
    results = ab_manager.complete_experiment(experiment_id)
    if not results:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return results


@app.post("/api/ab/experiments/{experiment_id}/pause")
async def pause_ab_experiment(experiment_id: str):
    """Pause an A/B experiment."""
    if not ab_manager.pause_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found or not active")
    return {"status": "paused"}


@app.post("/api/ab/experiments/{experiment_id}/resume")
async def resume_ab_experiment(experiment_id: str):
    """Resume a paused A/B experiment."""
    if not ab_manager.resume_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found or not paused")
    return {"status": "resumed"}


@app.delete("/api/ab/experiments/{experiment_id}")
async def delete_ab_experiment(experiment_id: str):
    """Delete an A/B experiment."""
    if not ab_manager.delete_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"status": "deleted"}


@app.post("/api/ab/assign/{experiment_id}")
async def assign_ab_group(experiment_id: str, client_id: str):
    """Assign a client to an A/B group."""
    try:
        group_name, group = ab_manager.assign_group(experiment_id, client_id)
        prom_metrics.ab_assignments_total.inc(experiment=experiment_id, group=group_name)
        return {
            "group": group_name,
            "model_version": group.model_version,
            "model_type": group.model_type,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    """Train universal model at end of day"""
    logger.info("Starting end-of-day universal training...")
    
    # Run universal training instead of per-ticker
    import threading
    training_thread = threading.Thread(target=train_universal_task, daemon=False)
    training_thread.start()
    logger.info("Universal training started in background")

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
