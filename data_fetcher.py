"""
Distributed Data Fetcher for Training Server
Fetches live market data for 135 stocks with caching and rate limiting
"""
import os
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import time
import logging
import io

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
        self.rate_limit_delay = 1.5  # seconds between requests (avoid 429)
        self.session = requests.Session()
        self._last_ping = None
        self._ping_ttl = 60  # seconds
        self.last_error = None
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3
        self.retry_delay = 5  # seconds

    def _yahoo_ping(self) -> str:
        """Check basic connectivity to Yahoo endpoints"""
        now = time.time()
        if self._last_ping and (now - self._last_ping) < self._ping_ttl:
            return "ok"
        try:
            resp = self.session.get(
                "https://query1.finance.yahoo.com/v7/finance/quote?symbols=AAPL",
                timeout=5
            )
            if resp.status_code == 200:
                self._last_ping = now
                return "ok"
            if resp.status_code == 429:
                self.last_error = "Yahoo rate limited (HTTP 429)"
                self.logger.warning(self.last_error)
                return "rate_limited"
            self.last_error = f"Yahoo ping failed: HTTP {resp.status_code}"
            self.logger.error(self.last_error)
            return "down"
        except Exception as e:
            self.last_error = f"Yahoo ping error: {e}"
            self.logger.error(self.last_error)
            return "down"

    def _stooq_symbol(self, ticker: str) -> str:
        """Map ticker to Stooq symbol"""
        return f"{ticker.lower()}.us"

    def _fetch_from_stooq(self, ticker: str) -> pd.DataFrame:
        """Fetch historical data from Stooq (daily)"""
        symbol = self._stooq_symbol(ticker)
        url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                self.last_error = f"Stooq HTTP {resp.status_code} for {ticker}"
                self.logger.error(self.last_error)
                return None
            df = pd.read_csv(io.StringIO(resp.text))
            if df is None or df.empty:
                self.last_error = f"Stooq empty dataset for {ticker}"
                self.logger.error(self.last_error)
                return None
            # Normalize columns
            df.rename(columns={
                "Date": "Date",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            }, inplace=True)
            return df
        except Exception as e:
            self.last_error = f"Stooq fetch error for {ticker}: {e}"
            self.logger.error(self.last_error)
            return None
        
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
        self.logger.info(f"Fetching {ticker}...")
        time.sleep(self.rate_limit_delay)

        ping_status = self._yahoo_ping()
        if ping_status == "down":
            self.logger.warning(f"Yahoo down - using Stooq for {ticker}")
            df = self._fetch_from_stooq(ticker)
            if df is not None and not df.empty:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                return df
            return None
        if ping_status == "rate_limited":
            self.logger.warning(f"Yahoo rate limited - using Stooq for {ticker}")
            df = self._fetch_from_stooq(ticker)
            if df is not None and not df.empty:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                return df
            self.logger.warning(f"Stooq failed for {ticker}, backing off Yahoo")
            time.sleep(self.retry_delay)
        
        try:
            df = None
            for attempt in range(1, self.max_retries + 1):
                df = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    session=self.session
                )
                if df is None or df.empty or len(df) < 200:
                    # Retry with longer period
                    df = yf.download(
                        ticker,
                        period="5y",
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                        session=self.session
                    )
                if df is None or df.empty or len(df) < 200:
                    # Final fallback
                    try:
                        df = yf.Ticker(ticker, session=self.session).history(period="max", auto_adjust=True)
                    except Exception:
                        df = None

                if df is not None and not df.empty:
                    break

                self.logger.warning(f"Retry {attempt}/{self.max_retries} for {ticker}")
                time.sleep(self.retry_delay * attempt)

            if df is None or df.empty:
                # Final fallback to Stooq
                self.logger.warning(f"Yahoo failed - fallback to Stooq for {ticker}")
                df = self._fetch_from_stooq(ticker)
            if df is None or df.empty:
                self.last_error = f"Error fetching {ticker}: empty dataset"
                self.logger.error(self.last_error)
                return None
            df = df.reset_index()
            if len(df) < 100:
                self.logger.warning(f"Warning: {ticker} data length is low ({len(df)} rows)")
            
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            return df
        except Exception as e:
            self.last_error = f"Error fetching {ticker}: {e}"
            self.logger.error(self.last_error)
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
