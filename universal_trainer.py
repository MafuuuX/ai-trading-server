"""
Universal Market Model Trainer – Server Edition  (v4 – optimised binary)
=============================================================================
Key improvements over v3
  • 5-day prediction horizon (much better SNR than 1-day)
  • Simplified architecture: Conv1D → single GRU → Dense (less overfitting)
  • Removed redundant features (Stoch_D, Williams_R, CCI) → 17 features
  • Removed regression head (conflicting gradients with classification)
  • Label smoothing 0.1 (reduces overconfidence on noisy labels)
  • Conv1D(32) feature extractor before GRU (local pattern detection)
  • Binary classification (UP / DOWN) with percentile labeling
  • Train-only normalisation (no data leakage)
  • 70/15/15 chronological split, class-weight balancing
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
import yfinance as yf
from datetime import datetime
from pathlib import Path

from data_fetcher import TOP_STOCKS

logger = logging.getLogger(__name__)

UNIVERSAL_MODEL_FILE = "models/universal_market_model.h5"
UNIVERSAL_SCALER_FILE = "models/universal_scaler.pkl"

# Prediction horizon: 5-day returns (much better signal-to-noise than 1-day)
PREDICTION_HORIZON = 5

# The exact feature list – MUST match universal_predictor.py & feature_engineering.py
# v4: removed Stoch_D (redundant with Stoch_K), Williams_R (≈inverted RSI), CCI (≈RSI)
FEATURE_COLS = [
    'Close_pct', 'High_pct', 'Low_pct', 'Volume_pct',
    'RSI', 'MACD_norm',
    'BB_position', 'BB_width',
    'ATR_pct',
    'SMA20_dist', 'SMA50_dist',
    'Stoch_K',
    'ROC_10',
    'OBV_pct',
    'Momentum',
    'VIX_level', 'VIX_change',
]


class UniversalModelTrainer:
    """Trains a single universal model across all tickers (binary UP/DOWN classification)"""

    def __init__(self, lookback: int = 20, epochs: int = 100):
        self.lookback = lookback
        self.epochs = epochs
        self.model = None
        self.scalers = {}  # ticker -> {mean, std}

    # ------------------------------------------------------------------
    # Feature engineering  (percentage / ratio only – no absolute prices)
    # ------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame, ticker: str, vix_df: pd.DataFrame = None):
        """
        Returns (X, (y_class, y_reg))  or  (None, None)
        All features are percentage-based or bounded [0-100] so that they
        generalise across tickers.
        """
        if df is None or df.empty or len(df) < self.lookback + 60:
            return None, None

        df = df.copy()

        close = df['Close']
        high  = df['High']
        low   = df['Low']
        vol   = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

        # ---- Percentage changes ----
        df['Close_pct']  = close.pct_change() * 100
        df['High_pct']   = high.pct_change() * 100
        df['Low_pct']    = low.pct_change() * 100
        df['Volume_pct'] = vol.pct_change() * 100

        # ---- RSI ----
        df['RSI'] = ta.momentum.rsi(close, window=14)

        # ---- MACD normalised by price ----
        macd_diff = ta.trend.macd_diff(close)
        df['MACD_norm'] = (macd_diff / close) * 100  # as % of price

        # ---- Bollinger Bands → position (0-1) & width (%) ----
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = bb_upper - bb_lower
        bb_range = bb_range.replace(0, np.nan)
        df['BB_position'] = ((close - bb_lower) / bb_range) * 100   # 0-100
        df['BB_width']    = (bb_range / close) * 100                 # width as %

        # ---- ATR as % of price ----
        atr = ta.volatility.average_true_range(high, low, close, window=14)
        df['ATR_pct'] = (atr / close) * 100

        # ---- SMA distances as % ----
        sma20 = ta.trend.sma_indicator(close, window=20)
        sma50 = ta.trend.sma_indicator(close, window=50)
        df['SMA20_dist'] = ((close - sma20) / close) * 100
        df['SMA50_dist'] = ((close - sma50) / close) * 100

        # ---- Stochastic %K, %D ----
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # ---- Rate of Change 10-day ----
        df['ROC_10'] = ta.momentum.roc(close, window=10)

        # ---- Williams %R ----
        df['Williams_R'] = ta.momentum.williams_r(high, low, close, lbp=14)

        # ---- CCI ----
        df['CCI'] = ta.trend.cci(high, low, close, window=20)

        # ---- OBV change % ----
        obv = ta.volume.on_balance_volume(close, vol)
        obv_pct = obv.pct_change() * 100
        df['OBV_pct'] = obv_pct.clip(-50, 50)  # clip extreme OBV spikes

        # ---- Momentum (10-day % change) ----
        df['Momentum'] = close.pct_change(10) * 100

        # ---- VIX market regime features ----
        try:
            if vix_df is not None and not vix_df.empty:
                # Align VIX to ticker's index; handle length mismatches
                vix_close = vix_df['Close']
                if len(vix_close) >= len(df):
                    # Use the last len(df) VIX values
                    df['VIX_level'] = vix_close.iloc[-len(df):].values
                else:
                    # VIX shorter than ticker data: forward-fill from end
                    aligned = vix_close.reindex(df.index).ffill().bfill().fillna(20.0)
                    df['VIX_level'] = aligned
                df['VIX_change'] = df['VIX_level'].pct_change() * 100
                df['VIX_level'] = df['VIX_level'] / 20.0
            else:
                # Fallback: neutral values if VIX unavailable
                df['VIX_level'] = 1.0    # 1.0 = VIX at ~20 (normal)
                df['VIX_change'] = 0.0
        except Exception as e:
            logger.warning(f"[Universal] VIX feature error for {ticker}: {e}")
            df['VIX_level'] = 1.0
            df['VIX_change'] = 0.0

        # ---- Target: 5-day percentage change (better SNR than 1-day) ----
        df['Target_pct'] = close.pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON) * 100

        # Drop NaNs introduced by indicator warm-up
        df = df.dropna()
        if len(df) < self.lookback + 1:
            return None, None

        features = df[FEATURE_COLS].values

        # NOTE: Raw features are returned — normalisation happens AFTER
        # the train/val/test split to prevent data leakage.
        # Scaler is computed on training data only (see train() method).

        # Sanity check: detect NaN/Inf in raw features
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning(f"[Universal] NaN/Inf in raw features for {ticker}, replacing")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        targets_pct = df['Target_pct'].values

        # Build sequences from RAW features (normalisation deferred)
        X, y_target = [], []
        for i in range(len(features) - self.lookback):
            X.append(features[i:i + self.lookback])
            y_target.append(targets_pct[i + self.lookback])

        return np.array(X), np.array(y_target)

    # ------------------------------------------------------------------
    # Model architecture  (v4: Conv1D → GRU → Dense, classification only)
    # ------------------------------------------------------------------
    def build_model(self, input_shape, learning_rate: float = 5e-4):
        inputs = layers.Input(shape=input_shape)

        # --- Conv1D feature extractor (local pattern detection) ---
        x = layers.Conv1D(32, kernel_size=3, padding='causal', activation=None)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # --- Single GRU encoder (simpler = less overfitting on noisy data) ---
        x = layers.GRU(64, dropout=0.2)(x)  # returns last hidden state only
        x = layers.LayerNormalization()(x)

        # --- Dense head ---
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.25)(x)

        # --- Classification output: Binary (UP=1 / DOWN=0) ---
        class_out = layers.Dense(2, activation='softmax', name='classification')(x)

        model = models.Model(inputs=inputs, outputs=class_out)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'],
        )
        return model

    # ------------------------------------------------------------------
    # Compute class weights to handle imbalance
    # ------------------------------------------------------------------
    @staticmethod
    def _class_weights(y):
        from collections import Counter
        counts = Counter(y.tolist())
        total = len(y)
        n_classes = len(counts)
        weights = {}
        for cls, cnt in counts.items():
            weights[int(cls)] = total / (n_classes * cnt)
        return weights

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, fetcher, tickers=None, progress_callback=None, fresh=False):
        """Train universal model.  If fresh=False (default) and a previous
        model exists on disk, it will be loaded and fine-tuned (warm start)
        with a lower learning rate.  Set fresh=True to train from scratch."""
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
                df = fetcher.fetch_historical_data(ticker, period="5y")
                if df is not None and len(df) >= 120:
                    all_data[ticker] = df
            except Exception as e:
                logger.warning(f"[Universal] Skip {ticker}: {e}")
            _progress("fetching", int((i + 1) / total_tickers * 30),
                       f"Fetched {ticker} ({i+1}/{total_tickers})")

        if len(all_data) < 5:
            logger.error(f"[Universal] Not enough data: only {len(all_data)} tickers (need >= 5)")
            _progress("error", 0, f"Failed: only {len(all_data)} tickers loaded (need 5+)")
            return None

        # --- Fetch VIX data for market regime features ---
        vix_df = None
        try:
            vix_raw = yf.download("^VIX", period="5y", progress=False)
            if vix_raw is not None and not vix_raw.empty:
                vix_raw = vix_raw.reset_index()
                if isinstance(vix_raw.columns, pd.MultiIndex):
                    vix_raw.columns = vix_raw.columns.droplevel(1)
                if 'Close' not in vix_raw.columns:
                    logger.warning("[Universal] VIX data missing 'Close' column")
                else:
                    vix_df = vix_raw[['Close']].copy()
                    vix_df.index = range(len(vix_df))
                    vix_median = vix_df['Close'].median()
                    if vix_median < 1 or vix_median > 200:
                        logger.warning(f"[Universal] VIX median={vix_median:.1f} looks wrong, ignoring")
                        vix_df = None
                    else:
                        logger.info(f"[Universal] VIX data loaded: {len(vix_df)} days, median={vix_median:.1f}")
            else:
                logger.warning("[Universal] VIX download returned empty data")
        except Exception as e:
            logger.warning(f"[Universal] VIX fetch failed (using fallback): {e}")

        # --- Phase 2: Prepare features (30-50%) ---
        _progress("features", 30, "Preparing features...")
        X_all, y_target_all = [], []
        ticker_indices = []  # track which samples belong to which ticker
        processed = 0
        for ticker, df in all_data.items():
            result = self.prepare_features(df, ticker, vix_df=vix_df)
            if result[0] is not None:
                X, y_target = result
                X_all.append(X)
                y_target_all.append(y_target)
                ticker_indices.append(len(X))
                processed += 1
            _progress("features", 30 + int(processed / len(all_data) * 20),
                       f"Features for {ticker} ({processed}/{len(all_data)})")

        if not X_all:
            logger.error("[Universal] No feature data produced from any ticker")
            _progress("error", 30, "Failed: no features could be computed")
            return None

        if len(X_all) < 3:
            logger.warning(f"[Universal] Only {len(X_all)} tickers produced features (low diversity)")

        # --- CHRONOLOGICAL 70/15/15 split per ticker ---
        # Train on oldest 70%, validate on next 15%, test on newest 15%
        X_train_all, X_val_all, X_test_all = [], [], []
        y_train_all, y_val_all, y_test_all = [], [], []

        for X_t, y_t in zip(X_all, y_target_all):
            if len(X_t) < 20:
                logger.warning(f"[Universal] Skipping ticker with only {len(X_t)} samples")
                continue
            split_train = int(0.70 * len(X_t))
            split_val = int(0.85 * len(X_t))
            if split_train < 10 or split_val - split_train < 3 or len(X_t) - split_val < 3:
                logger.warning(f"[Universal] Split too small ({split_train}/{split_val}/{len(X_t)}), skipping")
                continue
            X_train_all.append(X_t[:split_train])
            X_val_all.append(X_t[split_train:split_val])
            X_test_all.append(X_t[split_val:])
            y_train_all.append(y_t[:split_train])
            y_val_all.append(y_t[split_train:split_val])
            y_test_all.append(y_t[split_val:])

        if not X_train_all:
            logger.error("[Universal] No ticker data survived chronological split")
            _progress("error", 40, "Failed: all splits too small")
            return None

        X_train_raw = np.concatenate(X_train_all)
        X_val_raw   = np.concatenate(X_val_all)
        X_test_raw  = np.concatenate(X_test_all)
        y_train_raw = np.concatenate(y_train_all)
        y_val_raw   = np.concatenate(y_val_all)
        y_test_raw  = np.concatenate(y_test_all)

        # --- NORMALISATION: compute stats on TRAINING data only (no leakage) ---
        # Flatten sequences to (n_samples * lookback, n_features) for stats
        train_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])
        global_mean = train_flat.mean(axis=0)
        global_std  = train_flat.std(axis=0) + 1e-8

        def _normalize(X_raw):
            return np.clip((X_raw - global_mean) / global_std, -5.0, 5.0)

        X_train = _normalize(X_train_raw)
        X_val   = _normalize(X_val_raw)
        X_test  = _normalize(X_test_raw)

        # Save global scaler for all tickers (to be used by predictor)
        self.scalers = {'__global__': {'mean': global_mean.tolist(), 'std': global_std.tolist()}}
        # Also save per-ticker alias so predictor can find it
        for ticker in all_data.keys():
            self.scalers[ticker] = self.scalers['__global__']

        # Sanity: replace any remaining NaN/Inf
        for arr in (X_train, X_val, X_test):
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                arr[np.isnan(arr)] = 0.0
                arr[np.isinf(arr)] = 0.0

        # --- BINARY CLASSIFICATION with percentile-based labeling ---
        # Use median split: above median → UP (1), below median → DOWN (0)
        # This guarantees ~50/50 class balance
        train_median = np.median(y_train_raw)
        logger.info(f"[Universal] Percentile labeling: median {PREDICTION_HORIZON}-day return = {train_median:.4f}%")

        y_cls_train = keras.utils.to_categorical((y_train_raw > train_median).astype(np.int32), 2)
        y_cls_val   = keras.utils.to_categorical((y_val_raw > train_median).astype(np.int32), 2)
        y_cls_test  = keras.utils.to_categorical((y_test_raw > train_median).astype(np.int32), 2)

        logger.info(f"[Universal] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape} "
                     f"({processed} tickers, 70/15/15 chrono split)")
        logger.info(f"[Universal] Class balance train: DOWN={np.sum(y_cls_train[:,0]==1)}, UP={np.sum(y_cls_train[:,1]==1)}")

        # Shuffle ONLY the training data (val/test stay in order)
        idx = np.random.permutation(len(X_train))
        X_train     = X_train[idx]
        y_cls_train = y_cls_train[idx]

        # --- Class weights → sample weights ---
        y_cls_int = np.argmax(y_cls_train, axis=1)  # back to int for class weights
        cw = self._class_weights(y_cls_int)
        logger.info(f"[Universal] Class weights: {cw}")

        sample_weights = np.array([cw[int(c)] for c in y_cls_int])

        # --- Warm start: load previous model if available ---
        warm_start = False
        if not fresh and os.path.exists(UNIVERSAL_MODEL_FILE):
            try:
                self.model = keras.models.load_model(UNIVERSAL_MODEL_FILE, compile=False)
                # Validate feature dimensions match
                expected_shape = (self.lookback, len(FEATURE_COLS))
                model_input_shape = tuple(self.model.input_shape[1:])
                if model_input_shape != expected_shape:
                    logger.warning(
                        f"[Universal] Model shape mismatch: model={model_input_shape}, "
                        f"expected={expected_shape}. Training fresh instead."
                    )
                    self.model = None
                else:
                    # Re-compile with a lower LR for fine-tuning
                    self.model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                        metrics=['accuracy'],
                    )
                    warm_start = True
                    logger.info("[Universal] ♻️  Warm start – fine-tuning existing model (LR=1e-4)")
                    _progress("training", 50, "Warm start – fine-tuning existing model...")
            except Exception as e:
                logger.warning(f"[Universal] Could not load previous model, training fresh: {e}")
                self.model = None

        if self.model is None:
            self.model = self.build_model(X_train.shape[1:], learning_rate=5e-4)
            logger.info("[Universal] 🆕 Fresh training from scratch (LR=5e-4)")
            _progress("training", 50, "Training new model from scratch...")

        class EpochProgress(keras.callbacks.Callback):
            def on_epoch_end(cb_self, epoch, logs=None):
                pct = 50 + int((epoch + 1) / self.epochs * 45)
                acc = logs.get('val_accuracy', 0)
                _progress("training", pct,
                           f"Epoch {epoch+1}/{self.epochs} – "
                           f"acc={acc:.2%}")

        _progress("training", 50, "Training model...")
        try:
            history = self.model.fit(
                X_train,
                y_cls_train,
                validation_data=(X_val, y_cls_val),
                epochs=self.epochs,
                batch_size=64,
                sample_weight=sample_weights,
                shuffle=True,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=15, restore_best_weights=True,
                        mode='max',
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        patience=7, factor=0.5, min_lr=1e-6,
                    ),
                    EpochProgress(),
                ],
                verbose=0,
            )
        except Exception as e:
            logger.error(f"[Universal] Training crashed: {e}")
            _progress("error", 50, f"Training failed: {str(e)[:100]}")
            return None

        # --- Phase 4: Evaluate & save (95-100%) ---
        _progress("saving", 95, "Evaluating on test set...")
        try:
            # Evaluate on HELD-OUT test set (never seen during training/early-stopping)
            test_loss, cls_acc = self.model.evaluate(X_test, y_cls_test, verbose=0)
            # Also evaluate on validation set for comparison
            _, val_acc = self.model.evaluate(X_val, y_cls_val, verbose=0)
            logger.info(f"[Universal] Val accuracy={val_acc:.2%}, Test accuracy={cls_acc:.2%}")
            if val_acc - cls_acc > 0.05:
                logger.warning(f"[Universal] ⚠️ Val-Test gap ({val_acc:.2%} vs {cls_acc:.2%}) "
                               f"suggests overfitting to validation data")
        except Exception as e:
            logger.error(f"[Universal] Evaluation failed: {e}")
            cls_acc = 0.0
            val_acc = 0.0

        os.makedirs("models", exist_ok=True)

        # Backup previous model
        model_path = Path(UNIVERSAL_MODEL_FILE)
        if model_path.exists():
            backup_dir = Path("models/backups/universal")
            backup_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                model_path.rename(backup_dir / f"universal_model_{ts}.h5")
            except OSError as e:
                logger.warning(f"[Universal] Could not backup old model: {e}")

        try:
            self.model.save(UNIVERSAL_MODEL_FILE)
            # Save scaler + metadata (median threshold, classification type, lookback)
            scaler_data = {
                'scalers': self.scalers,
                'classification_type': 'binary',
                'label_median_threshold': float(train_median),
                'prediction_horizon': PREDICTION_HORIZON,
                'lookback': self.lookback,
                'n_features': len(FEATURE_COLS),
            }
            with open(UNIVERSAL_SCALER_FILE, 'wb') as f:
                pickle.dump(scaler_data, f)
        except Exception as e:
            logger.error(f"[Universal] Could not save model: {e}")
            _progress("error", 95, f"Model save failed: {e}")
            return None

        mode_label = "fine-tuned" if warm_start else "fresh"
        logger.info(f"[Universal] \u2705 test_accuracy={cls_acc:.2%} ({mode_label})")
        _progress("done", 100, f"Done \u2013 test_accuracy={cls_acc:.2%} ({mode_label})")

        return {
            "class_accuracy": float(cls_acc),
            "val_accuracy": float(val_acc) if 'val_acc' in dir() else float(cls_acc),
            "total_samples": int(len(X_train) + len(X_val) + len(X_test)),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "tickers_used": processed,
            "epochs_run": len(history.history.get('loss', [])),
            "warm_start": warm_start,
            "classification_type": "binary",
            "prediction_horizon": PREDICTION_HORIZON,
            "lookback": self.lookback,
            "label_median_threshold": float(train_median),
            "trained_at": datetime.now().isoformat(),
        }
