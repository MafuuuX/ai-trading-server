"""
Distributed Training Server with FastAPI
Serves models, manages training, and enables zero-downtime updates
"""
import asyncio
import os
import json
import hashlib
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

from data_fetcher import CachedDataFetcher, TOP_STOCKS
from trainer import ModelTrainer

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
        self.training_queue = []  # list of tickers
        self.is_training = False
        self.training_history = []
        self.last_training = {}
        self.logs = []
        
        self.scheduler = BackgroundScheduler()
        self.fetcher = CachedDataFetcher()
        self.trainer = ModelTrainer()
        
        # Track startup time for uptime calculation
        self.startup_time = datetime.now()

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


@app.get("/api/logs")
async def get_logs():
    """Recent server logs"""
    return {"logs": state.logs}


@app.get("/api/queue")
async def get_queue():
    """Current training queue"""
    return {"queue": state.training_queue}

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
    
    logger.info(f"Rolled back {ticker} to latest backup")
    return {"status": "rolled_back", "ticker": ticker}

@app.get("/api/training-status", response_model=Dict[str, TrainingStatus])
async def training_status():
    """Get training status for all stocks"""
    status_dict = {}
    
    for ticker in TOP_STOCKS[:20]:  # Return status for first 20
        status_dict[ticker] = {
            "ticker": ticker,
            "status": state.training_status.get(ticker, "idle"),
            "progress": 0.0,
            "last_trained": state.last_training.get(ticker),
            "next_training": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    
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
        
        # Fetch data
        df = state.fetcher.fetch_historical_data(ticker)
        if df is None or len(df) < 100:
            state.training_status[ticker] = "failed"
            logger.error(f"Insufficient data for {ticker}")
            return
        
        # Train model
        result = state.trainer.train_and_validate(df, ticker)
        
        if result:
            state.training_status[ticker] = "completed"
            state.last_training[ticker] = datetime.now().isoformat()
            state.active_models[ticker] = "v1"
            logger.info(f"✅ Completed training for {ticker}")
        else:
            state.training_status[ticker] = "failed"
            logger.error(f"Training failed for {ticker}")
            
    except Exception as e:
        state.training_status[ticker] = "failed"
        logger.error(f"Error training {ticker}: {e}")


def process_training_queue():
    """Process queued trainings sequentially"""
    if state.is_training:
        return
    state.is_training = True
    try:
        while state.training_queue:
            ticker = state.training_queue.pop(0)
            state.training_status[ticker] = "training"
            train_stock_task(ticker)
    finally:
        state.is_training = False

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

async def train_end_of_day():
    """Train all models at end of day"""
    logger.info(f"Starting end-of-day training for {len(TOP_STOCKS)} stocks...")
    
    for ticker in TOP_STOCKS[:20]:  # Start with first 20 for testing
        if state.training_status.get(ticker) not in ("training", "queued"):
            state.training_status[ticker] = "queued"
            if ticker not in state.training_queue:
                state.training_queue.append(ticker)
    process_training_queue()

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
