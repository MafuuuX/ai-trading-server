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

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request, Response
from fastapi.responses import FileResponse, JSONResponse
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

# Basic auth for web UI
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")


def _is_authorized(auth_header: str) -> bool:
    scheme, param = get_authorization_scheme_param(auth_header)
    if scheme.lower() != "basic" or not param:
        return False
    try:
        decoded = base64.b64decode(param).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return False
    return secrets.compare_digest(username, ADMIN_USER) and secrets.compare_digest(password, ADMIN_PASS)


@app.middleware("http")
async def ui_basic_auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/ui") or path.startswith("/static"):
        auth_header = request.headers.get("Authorization", "")
        if not _is_authorized(auth_header):
            return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})
    return await call_next(request)

# Global state
class ServerState:
    def __init__(self):
        self.models_dir = Path("./models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_models = {}  # ticker -> model_version
        self.training_status = {}  # ticker -> status
        self.training_history = []
        self.last_training = {}
        
        self.scheduler = BackgroundScheduler()
        self.fetcher = CachedDataFetcher()
        self.trainer = ModelTrainer()
        
        # Track startup time for uptime calculation
        self.startup_time = datetime.now()

state = ServerState()

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
    
    if state.training_status.get(ticker) == "training":
        raise HTTPException(status_code=409, detail=f"{ticker} already training")
    
    state.training_status[ticker] = "training"
    background_tasks.add_task(train_stock_task, ticker)
    
    return {"status": "training started", "ticker": ticker}

async def train_stock_task(ticker: str):
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

@app.post("/api/train-batch")
async def train_batch(tickers: List[str] = None, background_tasks: BackgroundTasks = None):
    """Start training for multiple stocks"""
    if tickers is None:
        tickers = TOP_STOCKS[:10]  # Default first 10
    
    for ticker in tickers:
        if state.training_status.get(ticker) != "training":
            state.training_status[ticker] = "queued"
            if background_tasks:
                background_tasks.add_task(train_stock_task, ticker)
    
    return {"status": "batch training started", "count": len(tickers)}

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
        if state.training_status.get(ticker) != "training":
            try:
                await train_stock_task(ticker)
            except Exception as e:
                logger.error(f"Error in end-of-day training for {ticker}: {e}")

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
