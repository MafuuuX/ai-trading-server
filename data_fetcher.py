"""
Distributed Data Fetcher for Training Server
Fetches live market data for 135 stocks with caching and rate limiting
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import time

# Top 135 stocks across all market caps
TOP_STOCKS = [
    # Mega Cap (7)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    
    # Large Cap Tech (20)
    'GOOG', 'AVGO', 'ORCL', 'INTC', 'AMD', 'NFLX', 'CRM', 'ADBE',
    'IBM', 'CSCO', 'QCOM', 'TXN', 'SNPS', 'CDNS', 'MCHP', 'INTU',
    'SHOP', 'CRWD', 'SPLK', 'DDOG',
    
    # Finance (18)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'COIN', 'V',
    'MA', 'AXP', 'DIS', 'CME', 'CBOE', 'ICE', 'BKNG', 'PYPL', 'VOYA',
    
    # Healthcare (20)
    'UNH', 'JNJ', 'LLY', 'MRK', 'PFE', 'ABBV', 'TMO', 'AMGN', 'GILD',
    'REGN', 'VRTX', 'ILMN', 'DXCM', 'VEEV', 'INCY', 'WDAY', 'ADUS',
    'EW', 'MSCI', 'MASI',
    
    # Energy (12)
    'CVX', 'XOM', 'COP', 'MPC', 'PSX', 'VLO', 'HES', 'SLB', 'FANG',
    'EOG', 'MUR', 'AR',
    
    # Consumer (15)
    'MCD', 'SBUX', 'DKNG', 'NKE', 'LULU', 'ULTA', 'GPS', 'WMT', 'KO',
    'PEP', 'CMG', 'HD', 'LOW', 'TJX', 'RH',
    
    # ETFs (20)
    'SPY', 'QQQ', 'IWM', 'EEM', 'FXI', 'BABA', 'IGV', 'XLK', 'XLV',
    'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLRE', 'XLU', 'VCIT', 'VGIT',
    'BND', 'AGG',
    
    # Additional Quality Stocks (33)
    'ZERO', 'NOW', 'TEAM', 'MDB', 'OKTA', 'ROKU', 'PINS', 'ZS', 'BILL',
    'UPST', 'RBLX', 'ASML', 'ARM', 'SMCI', 'REVG', 'GTLB', 'PSTG', 'DELL',
    'NET', 'CLOUDFLARE', 'SNOW', 'DBX', 'FIGMA', 'DOCUSIGN', 'TWLO', 'TTD',
    'MNST', 'ANET', 'ENPH', 'RUN', 'LI', 'NIO', 'XP', 'BILI',
]

class CachedDataFetcher:
    """Fetches data with intelligent caching and rate limiting"""
    
    def __init__(self, cache_dir: str = './data/cache', cache_hours: int = 2):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hours = cache_hours
        self.rate_limit_delay = 0.5  # seconds between requests
        
    def _get_cache_file(self, ticker: str) -> Path:
        """Get cache file path for ticker"""
        return self.cache_dir / f"{ticker}_cache.pkl"
    
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data is still fresh"""
        cache_file = self._get_cache_file(ticker)
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.total_seconds() < (self.cache_hours * 3600)
    
    def fetch_historical_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch or return cached historical data"""
        cache_file = self._get_cache_file(ticker)
        
        # Check cache first
        if self._is_cache_valid(ticker):
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                if cached is not None and not cached.empty:
                    return cached
                # Cached data is empty -> refetch
                try:
                    cache_file.unlink()
                except Exception:
                    pass
        
        # Fetch fresh data
        print(f"Fetching {ticker}...")
        time.sleep(self.rate_limit_delay)
        
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty or len(df) < 200:
                # Retry with longer period
                df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty:
                print(f"Error fetching {ticker}: empty dataset")
                return None
            df = df.reset_index()
            
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            return df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
    
    def fetch_batch(self, tickers: list = None, period: str = "2y") -> dict:
        """Fetch data for multiple tickers"""
        if tickers is None:
            tickers = TOP_STOCKS[:10]  # Default to first 10
        
        results = {}
        for ticker in tickers:
            df = self.fetch_historical_data(ticker, period)
            if df is not None:
                results[ticker] = df
        
        return results
    
    def clear_cache(self):
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("Cache cleared")


if __name__ == "__main__":
    # Example usage
    fetcher = CachedDataFetcher()
    
    # Fetch first 10 stocks
    data = fetcher.fetch_batch(TOP_STOCKS[:10])
    print(f"Fetched data for {len(data)} stocks")
    
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} days of data")
