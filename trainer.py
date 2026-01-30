"""
Distributed Model Trainer for Server
Trains dual-head LSTM models for 135 stocks with continuous learning
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import ta
import joblib
from pathlib import Path
from datetime import datetime

class ModelTrainer:
    """Trains and manages dual-head LSTM models"""
    
    def __init__(self, model_dir: str = './models', lookback: int = 30):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.model_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.lookback = lookback
        self.scaler = None
        
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        df = df.copy()
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            df['Close'] = df.get('Adj Close', df.get('close'))
        if 'High' not in df.columns or 'Low' not in df.columns:
            df['High'] = df.get('High', df.get('high'))
            df['Low'] = df.get('Low', df.get('low'))
        if 'Volume' not in df.columns:
            df['Volume'] = df.get('Volume', df.get('volume', 1))
        
        # Technical Indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        macd = ta.trend.macd(df['Close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd
        
        bb = ta.volatility.bollinger_bands(df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.iloc[:, 0]
        df['BB_Low'] = bb.iloc[:, 1]
        
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Price changes
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Fill NaN values (using new pandas syntax, not deprecated method parameter)
        df = df.bfill().fillna(0)
        
        return df
    
    def _build_dataset(self, df: pd.DataFrame, test_size: float = 0.2):
        """Build training/validation datasets"""
        df = self._add_features(df)
        
        # Ensure Close column has no NaN/None values
        if df['Close'].isna().any():
            print(f"⚠️ Found {df['Close'].isna().sum()} NaN values in Close column, forward-filling")
            df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining rows with NaN in critical columns
        critical_cols = ['Close', 'High', 'Low']
        df = df.dropna(subset=critical_cols)
        
        if df.empty or len(df) < self.lookback + 1:
            print(f"⚠️ Not enough valid data after NaN removal: {len(df)} rows (need {self.lookback + 1})")
            return None
        
        # Feature columns
        feature_cols = ['Close', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR', 
                       'SMA_20', 'SMA_50', 'Returns', 'Volatility', 'High', 'Low']
        
        # Ensure all feature columns exist and have no NaN
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing feature columns: {missing_cols}")
            return None
        
        # Fill remaining NaN values in feature columns before scaling
        for col in feature_cols:
            if df[col].isna().any():
                print(f"  Filling NaN in {col}")
                df[col] = df[col].ffill().bfill()
                # If still NaN (e.g., at start), use 0
                df[col] = df[col].fillna(0)
        
        # Final check: drop rows with any NaN in features
        df = df.dropna(subset=feature_cols)
        
        if len(df) < self.lookback + 1:
            print(f"⚠️ Insufficient data after feature validation: {len(df)} rows")
            return None
        
        # Reset index again after dropping rows
        df = df.reset_index(drop=True)
        
        X = df[feature_cols].values
        
        # Verify X has no NaN before scaling
        if np.any(np.isnan(X)):
            nan_count = np.isnan(X).sum()
            print(f"⚠️ ERROR: X still contains {nan_count} NaN values after preprocessing!")
            return None
        
        # Scale features
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_class, y_reg = [], [], []
        for i in range(len(X_scaled) - self.lookback):
            # Classification: 0=DOWN, 1=NEUTRAL, 2=UP
            close_current = df['Close'].iloc[i]
            close_future = df['Close'].iloc[i+self.lookback]
            
            # Defensive check for NaN/None
            if pd.isna(close_current) or pd.isna(close_future):
                continue
            
            future_return = (close_future - close_current) / close_current
            if pd.isna(future_return) or not np.isfinite(future_return):
                continue
            
            # Only add sequence if target is valid
            X_seq.append(X_scaled[i:i+self.lookback])
            
            if future_return < -0.01:
                label = 0  # DOWN
            elif future_return > 0.01:
                label = 2  # UP
            else:
                label = 1  # NEUTRAL
            y_class.append(label)
            y_reg.append(future_return)
        
        if len(X_seq) < 50:
            print(f"⚠️ Not enough valid sequences ({len(X_seq)}, need 50+)")
            return None
        
        X_seq = np.array(X_seq)
        y_class = keras.utils.to_categorical(np.array(y_class), 3)
        y_reg = np.array(y_reg).reshape(-1, 1)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train = X_seq[:split_idx]
        y_class_train = y_class[:split_idx]
        y_reg_train = y_reg[:split_idx]
        
        X_val = X_seq[split_idx:]
        y_class_val = y_class[split_idx:]
        y_reg_val = y_reg[split_idx:]
        
        return X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val
    
    def _create_model(self, input_shape: tuple) -> Model:
        """Create dual-head LSTM model"""
        inputs = Input(shape=input_shape)
        
        # Shared LSTM layers
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        
        # Classification head
        class_x = layers.Dense(16, activation='relu')(x)
        class_x = layers.Dropout(0.1)(class_x)
        class_out = layers.Dense(3, activation='softmax', name='classifier')(class_x)
        
        # Regression head
        reg_x = layers.Dense(16, activation='relu')(x)
        reg_x = layers.Dropout(0.1)(reg_x)
        reg_out = layers.Dense(1, activation='linear', name='regressor')(reg_x)
        
        model = Model(inputs=inputs, outputs=[class_out, reg_out])
        return model
    
    def train_and_validate(self, df: pd.DataFrame, ticker: str, epochs: int = 50, progress_callback=None) -> dict:
        """Train model and return validation metrics"""
        print(f"\nTraining {ticker}...")
        
        # Build dataset
        result = self._build_dataset(df)
        if result is None:
            print(f"⚠️ Failed to build dataset for {ticker}")
            return None
        
        X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val = result
        
        if len(X_train) < 50:
            print(f"⚠️ Insufficient data for {ticker} ({len(X_train)} samples)")
            return None
        
        # Create model
        model = self._create_model((X_train.shape[1], X_train.shape[2]))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'classifier': 'categorical_crossentropy',
                'regressor': 'mse'
            },
            metrics={
                'classifier': 'accuracy',
                'regressor': 'mae'
            },
            loss_weights={'classifier': 0.7, 'regressor': 0.3}
        )
        
        # Progress callback
        if progress_callback:
            class ProgressCallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    try:
                        progress_callback(epoch + 1, epochs)
                    except Exception:
                        pass
            progress_cb = ProgressCallback()
        else:
            progress_cb = None

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_classifier_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        if progress_cb:
            callbacks.append(progress_cb)
        
        # Train
        history = model.fit(
            X_train, [y_class_train, y_reg_train],
            validation_data=(X_val, [y_class_val, y_reg_val]),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        val_results = model.evaluate(X_val, [y_class_val, y_reg_val], verbose=0)
        
        # Save model and scaler (with backup)
        model_path = self.model_dir / f"{ticker}_model.h5"
        scaler_path = self.model_dir / f"{ticker}_scaler.pkl"

        # Backup existing files
        if model_path.exists() or scaler_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ticker_backup_dir = self.backup_dir / ticker
            ticker_backup_dir.mkdir(parents=True, exist_ok=True)
            
            if model_path.exists():
                backup_model = ticker_backup_dir / f"{ticker}_model_{ts}.h5"
                model_path.replace(backup_model)
            if scaler_path.exists():
                backup_scaler = ticker_backup_dir / f"{ticker}_scaler_{ts}.pkl"
                scaler_path.replace(backup_scaler)
        
        model.save(str(model_path))
        joblib.dump(self.scaler, str(scaler_path))
        
        print(f"✅ {ticker} trained - Val Loss: {val_results[0]:.4f}")
        
        return {
            'ticker': ticker,
            'val_loss': float(val_results[0]),
            'class_accuracy': float(val_results[2]) if len(val_results) > 2 else 0,
            'reg_mae': float(val_results[3]) if len(val_results) > 3 else 0,
            'trained_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example: Train single stock
    from data_fetcher import CachedDataFetcher
    
    fetcher = CachedDataFetcher()
    trainer = ModelTrainer()
    
    df = fetcher.fetch_historical_data('AAPL')
    if df is not None:
        result = trainer.train_and_validate(df, 'AAPL')
        print(f"Result: {result}")
