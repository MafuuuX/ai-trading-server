"""
Debug script to identify NoneType issue in data pipeline
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from data_fetcher import CachedDataFetcher
from trainer import ModelTrainer

# Test data fetching
print("=" * 60)
print("Testing Data Fetcher for AAPL")
print("=" * 60)

fetcher = CachedDataFetcher()
df = fetcher.fetch_historical_data("AAPL", period="2y")

if df is None:
    print("❌ Data fetch returned None")
    sys.exit(1)

print(f"\n✓ Data shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")
print(f"✓ Dtypes:\n{df.dtypes}")

# Check for NaN values
print(f"\n📊 NaN Analysis:")
nan_counts = df.isna().sum()
if nan_counts.any():
    print(f"❌ Found NaN values:")
    print(nan_counts[nan_counts > 0])
else:
    print("✓ No NaN values found")

# Check data types
print(f"\n🔍 Data Type Analysis:")
for col in ["Close", "Open", "High", "Low"]:
    if col in df.columns:
        val = df[col].iloc[0]
        print(f"  {col}: {type(val)} = {val}")
        # Try arithmetic
        try:
            val_current = df[col].iloc[10]
            val_future = df[col].iloc[11]
            result = val_future - val_current
            print(f"    Arithmetic test: {val_future} - {val_current} = {result} ✓")
        except Exception as e:
            print(f"    Arithmetic test FAILED: {e} ❌")

# Check for None values specifically
print(f"\n🔎 Explicit None Check:")
for col in ["Close", "Open", "High", "Low"]:
    if col in df.columns:
        none_count = df[col].apply(lambda x: x is None).sum()
        if none_count > 0:
            print(f"  ❌ {col} has {none_count} None values")
        else:
            print(f"  ✓ {col} has no None values")

# Sample data
print(f"\n📋 First 10 rows:")
print(df[["Date", "Open", "High", "Low", "Close"]].head(10))

# Test trainer
print("\n" + "=" * 60)
print("Testing Trainer Dataset Building")
print("=" * 60)

trainer = ModelTrainer()
result = trainer._build_dataset(df)

if result is None:
    print("❌ _build_dataset returned None")
    sys.exit(1)

X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val = result

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ y_class_train shape: {y_class_train.shape}")
print(f"✓ y_reg_train shape: {y_reg_train.shape}")

print("\n✅ All tests passed!")
