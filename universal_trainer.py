"""
Universal Market Model Trainer - Server Edition
Trainiert ein großes Modell mit allen TOP_STOCKS gleichzeitig
für universelle Aktien-Kurs-Vorhersagen.

Verwendet die Server-eigene CachedDataFetcher Instanz.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import ta
import pickle
import os
import logging
from datetime import datetime
from pathlib import Path

from data_fetcher import TOP_STOCKS

logger = logging.getLogger(__name__)

UNIVERSAL_MODEL_FILE = "models/universal_market_model.h5"
UNIVERSAL_SCALER_FILE = "models/universal_scaler.pkl"


class UniversalModelTrainer:
    """Trains a single universal model across all tickers"""

    def __init__(self, lookback: int = 30, epochs: int = 50):
        self.lookback = lookback
        self.epochs = epochs
        self.model = None
        self.scalers = {}  # ticker -> {mean, std}

    # ------------------------------------------------------------------
    # Feature engineering (percentage-based, ticker-agnostic)
    # ------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame, ticker: str):
        """
        Prepare features: % changes + technical indicators.

        Returns (X, (y_class, y_reg))  or  (None, None)
        """
        if df is None or df.empty or len(df) < self.lookback + 50:
            return None, None

        df = df.copy()

        # Percentage changes (ticker-agnostic)
        df['Close_pct'] = df['Close'].pct_change() * 100
        df['High_pct'] = df['High'].pct_change() * 100
        df['Low_pct'] = df['Low'].pct_change() * 100
        if 'Volume' in df.columns:
            df['Volume_pct'] = df['Volume'].pct_change() * 100
        else:
            df['Volume_pct'] = 0.0

        # Technical indicators via ta library
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])

        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()

        atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['ATR_pct'] = (atr / df['Close']) * 100

        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)

        # Target: next-day percentage change
        df['Target_pct'] = df['Close'].pct_change(1).shift(-1) * 100

        df = df.dropna()
        if len(df) < self.lookback + 1:
            return None, None

        feature_cols = [
            'Close_pct', 'High_pct', 'Low_pct', 'Volume_pct',
            'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower',
            'ATR_pct', 'SMA_20', 'SMA_50',
        ]

        features = df[feature_cols].values

        # Z-score normalisation per ticker
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features_norm = (features - mean) / std
        self.scalers[ticker] = {'mean': mean.tolist(), 'std': std.tolist()}

        targets_pct = df['Target_pct'].values
        targets_class = np.array([
            2 if p > 0.6 else (0 if p < -0.6 else 1)
            for p in targets_pct
        ])
        targets_reg = targets_pct

        X, y_cls, y_reg = [], [], []
        for i in range(len(features_norm) - self.lookback):
            X.append(features_norm[i:i + self.lookback])
            y_cls.append(targets_class[i + self.lookback])
            y_reg.append(targets_reg[i + self.lookback])

        return np.array(X), (np.array(y_cls), np.array(y_reg))

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    def build_model(self, input_shape):
        """Bidirectional GRU dual-head model"""
        inputs = layers.Input(shape=input_shape)

        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=True, dropout=0.2)
        )(inputs)
        x = layers.GRU(32, dropout=0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        class_out = layers.Dense(3, activation='softmax', name='classification')(x)
        reg_out = layers.Dense(1, name='regression')(x)

        model = models.Model(inputs=inputs, outputs=[class_out, reg_out])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'regression': 'mse',
            },
            metrics={
                'classification': 'accuracy',
                'regression': 'mae',
            },
            loss_weights={'classification': 1.0, 'regression': 0.5},
        )
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, fetcher, tickers=None, progress_callback=None):
        """
        Train the universal model.

        Args:
            fetcher: CachedDataFetcher instance (server-side)
            tickers: list of tickers (defaults to TOP_STOCKS)
            progress_callback: fn(phase, progress_pct, message)

        Returns:
            dict with metrics on success, None on failure
        """
        tickers = tickers or TOP_STOCKS
        total_tickers = len(tickers)

        def _progress(phase, pct, msg=""):
            if progress_callback:
                progress_callback(phase, pct, msg)

        # --- Phase 1: Fetch data (0-30%) ---
        _progress("fetching", 0, f"Fetching data for {total_tickers} tickers...")
        all_data = {}
        for i, ticker in enumerate(tickers):
            try:
                df = fetcher.fetch_historical_data(ticker, period="2y")
                if df is not None and len(df) >= 100:
                    all_data[ticker] = df
            except Exception as e:
                logger.warning(f"[Universal] Skip {ticker}: {e}")
            _progress("fetching", int((i + 1) / total_tickers * 30),
                       f"Fetched {ticker} ({i+1}/{total_tickers})")

        if len(all_data) < 5:
            logger.error(f"[Universal] Not enough data: only {len(all_data)} tickers")
            return None

        # --- Phase 2: Prepare features (30-50%) ---
        _progress("features", 30, "Preparing features...")
        X_all, y_cls_all, y_reg_all = [], [], []
        processed = 0
        for ticker, df in all_data.items():
            result = self.prepare_features(df, ticker)
            if result[0] is not None:
                X, (y_cls, y_reg) = result
                X_all.append(X)
                y_cls_all.append(y_cls)
                y_reg_all.append(y_reg)
                processed += 1
            _progress("features", 30 + int(processed / len(all_data) * 20),
                       f"Features for {ticker} ({processed}/{len(all_data)})")

        if not X_all:
            logger.error("[Universal] No feature data produced")
            return None

        X_combined = np.concatenate(X_all)
        y_cls_combined = np.concatenate(y_cls_all)
        y_reg_combined = np.concatenate(y_reg_all)

        logger.info(f"[Universal] Combined dataset: {X_combined.shape} "
                     f"({processed} tickers)")

        # --- Phase 3: Train (50-95%) ---
        split = int(0.8 * len(X_combined))
        X_train, X_val = X_combined[:split], X_combined[split:]
        y_cls_train, y_cls_val = y_cls_combined[:split], y_cls_combined[split:]
        y_reg_train, y_reg_val = y_reg_combined[:split], y_reg_combined[split:]

        self.model = self.build_model(X_train.shape[1:])

        class EpochProgress(keras.callbacks.Callback):
            def on_epoch_end(cb_self, epoch, logs=None):
                pct = 50 + int((epoch + 1) / self.epochs * 45)
                acc = logs.get('val_classification_accuracy', 0)
                mae = logs.get('val_regression_mae', 0)
                _progress("training", pct,
                           f"Epoch {epoch+1}/{self.epochs} – "
                           f"acc={acc:.2%}, mae={mae:.4f}")

        _progress("training", 50, "Training model...")
        history = self.model.fit(
            X_train,
            {'classification': y_cls_train, 'regression': y_reg_train},
            validation_data=(
                X_val,
                {'classification': y_cls_val, 'regression': y_reg_val},
            ),
            epochs=self.epochs,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_classification_accuracy',
                    patience=10, restore_best_weights=True,
                    mode='max',
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=5, factor=0.5, min_lr=1e-6
                ),
                EpochProgress(),
            ],
            verbose=0,
        )

        # --- Phase 4: Evaluate & save (95-100%) ---
        _progress("saving", 95, "Evaluating & saving...")
        loss, cls_loss, reg_loss, cls_acc, reg_mae = self.model.evaluate(
            X_val,
            {'classification': y_cls_val, 'regression': y_reg_val},
            verbose=0,
        )

        os.makedirs("models", exist_ok=True)

        # Backup previous model
        model_path = Path(UNIVERSAL_MODEL_FILE)
        if model_path.exists():
            backup_dir = Path("models/backups/universal")
            backup_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path.rename(backup_dir / f"universal_model_{ts}.h5")

        self.model.save(UNIVERSAL_MODEL_FILE)
        with open(UNIVERSAL_SCALER_FILE, 'wb') as f:
            pickle.dump(self.scalers, f)

        logger.info(f"[Universal] ✅ Model saved – accuracy={cls_acc:.2%}, MAE={reg_mae:.4f}")
        _progress("done", 100, f"Done – accuracy={cls_acc:.2%}")

        return {
            "class_accuracy": float(cls_acc),
            "reg_mae": float(reg_mae),
            "total_samples": int(len(X_combined)),
            "tickers_used": processed,
            "epochs_run": len(history.history.get('loss', [])),
            "trained_at": datetime.now().isoformat(),
        }
