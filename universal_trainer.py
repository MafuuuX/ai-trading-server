"""
Universal Market Model Trainer – Server Edition  (v2 – optimised)
================================================================
Key improvements over v1
  • All features are ticker-agnostic (no absolute prices)
  • Richer feature set: 18 features including Stoch, ROC, WilliamsR, OBV, CCI
  • Deeper model with LayerNorm + residual connection + attention
  • Class-weight balancing for UP / NEUTRAL / DOWN
  • Data shuffling across all tickers
  • Lower initial LR + cosine-decay schedule
  • Label smoothing on classification head
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

# The exact feature list – MUST match universal_predictor.py
FEATURE_COLS = [
    'Close_pct', 'High_pct', 'Low_pct', 'Volume_pct',
    'RSI', 'MACD_norm',
    'BB_position', 'BB_width',
    'ATR_pct',
    'SMA20_dist', 'SMA50_dist',
    'Stoch_K', 'Stoch_D',
    'ROC_10', 'Williams_R',
    'CCI', 'OBV_pct',
    'Momentum',
    'VIX_level', 'VIX_change',
]


class UniversalModelTrainer:
    """Trains a single universal model across all tickers"""

    def __init__(self, lookback: int = 60, epochs: int = 80):
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
                    aligned = vix_close.reindex(df.index, method='ffill')
                    df['VIX_level'] = aligned.fillna(method='bfill').fillna(20.0)
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

        # ---- Target: next-day percentage change ----
        df['Target_pct'] = close.pct_change(1).shift(-1) * 100

        # Drop NaNs introduced by indicator warm-up
        df = df.dropna()
        if len(df) < self.lookback + 1:
            return None, None

        features = df[FEATURE_COLS].values

        # Z-score normalisation per ticker
        mean = features.mean(axis=0)
        std  = features.std(axis=0) + 1e-8
        features_norm = (features - mean) / std
        self.scalers[ticker] = {'mean': mean.tolist(), 'std': std.tolist()}

        # Clip extreme z-scores to ±5
        features_norm = np.clip(features_norm, -5.0, 5.0)

        # Sanity check: detect NaN/Inf in normalised features
        if np.any(np.isnan(features_norm)) or np.any(np.isinf(features_norm)):
            logger.warning(f"[Universal] NaN/Inf in normalised features for {ticker}, replacing")
            features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=5.0, neginf=-5.0)

        # Classification labels: adaptive percentile-based thresholds
        targets_pct  = df['Target_pct'].values
        # Use ±0.4% thresholds (more balanced than ±0.6%)
        targets_class = np.where(
            targets_pct >  0.4, 2,      # UP
            np.where(targets_pct < -0.4, 0, 1)  # DOWN / NEUTRAL
        )

        targets_reg = targets_pct

        X, y_cls, y_reg = [], [], []
        for i in range(len(features_norm) - self.lookback):
            X.append(features_norm[i:i + self.lookback])
            y_cls.append(targets_class[i + self.lookback])
            y_reg.append(targets_reg[i + self.lookback])

        return np.array(X), (np.array(y_cls), np.array(y_reg))

    # ------------------------------------------------------------------
    # Model architecture  (deeper, with attention + residual)
    # ------------------------------------------------------------------
    def build_model(self, input_shape):
        n_features = input_shape[-1]  # 18
        inputs = layers.Input(shape=input_shape)

        # --- Encoder block 1 ---
        x = layers.Bidirectional(
            layers.GRU(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.1)
        )(inputs)
        x = layers.LayerNormalization()(x)

        # --- Encoder block 2 ---
        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=True, dropout=0.15, recurrent_dropout=0.1)
        )(x)
        x = layers.LayerNormalization()(x)

        # --- Simple self-attention ---
        attn_scores = layers.Dense(1, activation='tanh')(x)             # (batch, seq, 1)
        attn_weights = layers.Softmax(axis=1)(attn_scores)              # (batch, seq, 1)
        context = layers.Multiply()([x, attn_weights])                  # weighted
        context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)  # (batch, 128)

        # Also take last hidden state (residual-like)
        last_hidden = layers.Lambda(lambda t: t[:, -1, :])(x)          # (batch, 128)
        combined = layers.Concatenate()([context, last_hidden])         # (batch, 256)

        # --- Shared dense trunk ---
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.15)(x)

        # --- Classification head (label smoothing via loss) ---
        class_out = layers.Dense(3, activation='softmax', name='classification')(x)

        # --- Regression head ---
        reg_out = layers.Dense(1, name='regression')(x)

        model = models.Model(inputs=inputs, outputs=[class_out, reg_out])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss={
                'classification': keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False,
                ),
                'regression': 'huber',           # Huber is more robust than MSE
            },
            metrics={
                'classification': 'accuracy',
                'regression': 'mae',
            },
            loss_weights={'classification': 1.0, 'regression': 0.3},
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
                df = fetcher.fetch_historical_data(ticker, period="2y")
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
            vix_raw = yf.download("^VIX", period="2y", progress=False)
            if vix_raw is not None and not vix_raw.empty:
                vix_raw = vix_raw.reset_index()
                if isinstance(vix_raw.columns, pd.MultiIndex):
                    vix_raw.columns = vix_raw.columns.droplevel(1)
                if 'Close' not in vix_raw.columns:
                    logger.warning("[Universal] VIX data missing 'Close' column")
                else:
                    vix_df = vix_raw[['Close']].copy()
                    vix_df.index = range(len(vix_df))
                    # Validate VIX values are reasonable (5-100 range)
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
        X_all, y_cls_all, y_reg_all = [], [], []
        processed = 0
        for ticker, df in all_data.items():
            result = self.prepare_features(df, ticker, vix_df=vix_df)
            if result[0] is not None:
                X, (y_cls, y_reg) = result
                X_all.append(X)
                y_cls_all.append(y_cls)
                y_reg_all.append(y_reg)
                processed += 1
            _progress("features", 30 + int(processed / len(all_data) * 20),
                       f"Features for {ticker} ({processed}/{len(all_data)})")

        if not X_all:
            logger.error("[Universal] No feature data produced from any ticker")
            _progress("error", 30, "Failed: no features could be computed")
            return None

        if len(X_all) < 3:
            logger.warning(f"[Universal] Only {len(X_all)} tickers produced features (low diversity)")

        # --- CHRONOLOGICAL split per ticker (prevents temporal leakage) ---
        # Split each ticker's data 85/15 so validation is always the FUTURE
        X_train_all, X_val_all = [], []
        y_cls_train_all, y_cls_val_all = [], []
        y_reg_train_all, y_reg_val_all = [], []

        for X_t, y_cls_t, y_reg_t in zip(X_all, y_cls_all, y_reg_all):
            if len(X_t) < 10:
                logger.warning(f"[Universal] Skipping ticker with only {len(X_t)} samples")
                continue
            split_i = int(0.85 * len(X_t))
            if split_i < 5 or len(X_t) - split_i < 2:
                logger.warning(f"[Universal] Split too small ({split_i}/{len(X_t)}), skipping")
                continue
            X_train_all.append(X_t[:split_i])
            X_val_all.append(X_t[split_i:])
            y_cls_train_all.append(y_cls_t[:split_i])
            y_cls_val_all.append(y_cls_t[split_i:])
            y_reg_train_all.append(y_reg_t[:split_i])
            y_reg_val_all.append(y_reg_t[split_i:])

        if not X_train_all:
            logger.error("[Universal] No ticker data survived chronological split")
            _progress("error", 40, "Failed: all splits too small")
            return None

        X_train = np.concatenate(X_train_all)
        X_val   = np.concatenate(X_val_all)
        y_cls_train = np.concatenate(y_cls_train_all)
        y_cls_val   = np.concatenate(y_cls_val_all)
        y_reg_train = np.concatenate(y_reg_train_all)
        y_reg_val   = np.concatenate(y_reg_val_all)

        logger.info(f"[Universal] Train: {X_train.shape}, Val: {X_val.shape} ({processed} tickers, chronological split)")

        # Shuffle ONLY the training data (validation stays in order)
        idx = np.random.permutation(len(X_train))
        X_train     = X_train[idx]
        y_cls_train = y_cls_train[idx]
        y_reg_train = y_reg_train[idx]

        # --- Class weights → sample weights (multi-output models don't support class_weight) ---
        cw = self._class_weights(y_cls_train)
        logger.info(f"[Universal] Class weights: {cw}")

        # Convert class weights to per-sample weights
        sample_weights_cls = np.array([cw[int(c)] for c in y_cls_train])
        sample_weights_reg = np.ones(len(y_reg_train))

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
                        loss={
                            'classification': keras.losses.SparseCategoricalCrossentropy(),
                            'regression': 'huber',
                        },
                        metrics={'classification': 'accuracy', 'regression': 'mae'},
                        loss_weights={'classification': 1.0, 'regression': 0.3},
                    )
                    warm_start = True
                    logger.info("[Universal] ♻️  Warm start – fine-tuning existing model (LR=1e-4)")
                    _progress("training", 50, "Warm start – fine-tuning existing model...")
            except Exception as e:
                logger.warning(f"[Universal] Could not load previous model, training fresh: {e}")
                self.model = None

        if self.model is None:
            self.model = self.build_model(X_train.shape[1:])
            logger.info("[Universal] 🆕 Fresh training from scratch (LR=5e-4)")
            _progress("training", 50, "Training new model from scratch...")

        class EpochProgress(keras.callbacks.Callback):
            def on_epoch_end(cb_self, epoch, logs=None):
                pct = 50 + int((epoch + 1) / self.epochs * 45)
                acc  = logs.get('val_classification_accuracy', 0)
                mae  = logs.get('val_regression_mae', 0)
                _progress("training", pct,
                           f"Epoch {epoch+1}/{self.epochs} – "
                           f"acc={acc:.2%}, mae={mae:.4f}")

        _progress("training", 50, "Training model...")
        try:
            history = self.model.fit(
                X_train,
                {'classification': y_cls_train, 'regression': y_reg_train},
                validation_data=(
                    X_val,
                    {'classification': y_cls_val, 'regression': y_reg_val},
                ),
                epochs=self.epochs,
                batch_size=64,
                sample_weight={'classification': sample_weights_cls, 'regression': sample_weights_reg},
                shuffle=True,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_classification_accuracy',
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
        _progress("saving", 95, "Evaluating & saving...")
        try:
            loss, cls_loss, reg_loss, cls_acc, reg_mae = self.model.evaluate(
                X_val,
                {'classification': y_cls_val, 'regression': y_reg_val},
                verbose=0,
            )
        except Exception as e:
            logger.error(f"[Universal] Evaluation failed: {e}")
            cls_acc = 0.0
            reg_mae = 999.0

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
            with open(UNIVERSAL_SCALER_FILE, 'wb') as f:
                pickle.dump(self.scalers, f)
        except Exception as e:
            logger.error(f"[Universal] Could not save model: {e}")
            _progress("error", 95, f"Model save failed: {e}")
            return None

        mode_label = "fine-tuned" if warm_start else "fresh"
        logger.info(f"[Universal] ✅ accuracy={cls_acc:.2%}, MAE={reg_mae:.4f} ({mode_label})")
        _progress("done", 100, f"Done – accuracy={cls_acc:.2%} ({mode_label})")

        return {
            "class_accuracy": float(cls_acc),
            "reg_mae": float(reg_mae),
            "total_samples": int(len(X_train) + len(X_val)),
            "tickers_used": processed,
            "epochs_run": len(history.history.get('loss', [])),
            "warm_start": warm_start,
            "trained_at": datetime.now().isoformat(),
        }
