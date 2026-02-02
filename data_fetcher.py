import yfinance as yf
import pandas as pd
import random
import requests
import time
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from pathlib import Path

# Top stocks for training
TOP_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'GOOG', 'NFLX', 'AMD',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'COIN', 'V', 'MA', 'AXP',
    'UNH', 'JNJ', 'LLY', 'MRK', 'PFE', 'ABBV', 'TMO', 'AMGN',
    'CVX', 'XOM', 'COP', 'MPC', 'PSX', 'VLO', 'HES', 'SLB',
    'MCD', 'SBUX', 'NKE', 'LULU', 'WMT', 'KO', 'PEP', 'HD',
    'SPY', 'QQQ', 'IWM', 'EEM', 'XLK', 'XLV', 'XLF', 'XLE',
    'AVGO', 'ORCL', 'INTC', 'CRM', 'ADBE', 'IBM', 'CSCO', 'QCOM', 'TXN'
]

# ============================================================================
# API CONFIGURATION
# ============================================================================
# Try to load API keys from parent directory's api_keys.py
try:
    sys.path.insert(0, str((Path(__file__).parent.parent).resolve()))
    from api_keys import load_api_keys
    API_KEYS = load_api_keys()
    print("[Server] API Keys aus api_keys.py geladen")
except ImportError:
    # Fallback to environment variables
    API_KEYS = {
        'finnhub': os.environ.get('FINNHUB_API_KEY', ''),
        'alpaca': os.environ.get('ALPACA_API_KEY_ID', ''),
        'alpaca_secret': os.environ.get('ALPACA_SECRET_KEY', ''),
        'alpaca_base_url': os.environ.get('ALPACA_BASE_URL', 'https://data.alpaca.markets'),
    }
    print("[Server] API Keys aus Umgebungsvariablen geladen")

# Rate limiting
_last_request_time: Dict[str, float] = {}
RATE_LIMITS = {
    'alpaca': 0,       # Alpaca hat keine Limits - parallel möglich
    'finnhub': 1.0,    # 1 Sekunde zwischen Anfragen
    'yahoo': 0.5,      # 2/Sekunde
}

def _rate_limit(provider: str):
    """Wartet falls nötig um Rate Limits einzuhalten"""
    if provider in _last_request_time:
        elapsed = time.time() - _last_request_time[provider]
        wait_time = RATE_LIMITS.get(provider, 0.5) - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
    _last_request_time[provider] = time.time()


# ============================================================================
# PROVIDER IMPLEMENTATIONS
# ============================================================================

def get_price_alpaca(ticker: str) -> Optional[float]:
    """Holt Live-Preis von Alpaca (IEX Feed, kostenlos, UNLIMITED)"""
    api_key = API_KEYS.get('alpaca', '')
    secret_key = API_KEYS.get('alpaca_secret', '')
    base_url = API_KEYS.get('alpaca_base_url', 'https://data.alpaca.markets')
    if not api_key or not secret_key:
        return None

    try:
        _rate_limit('alpaca')
        url = f"{base_url}/v2/stocks/{ticker}/quotes/latest"
        params = {"feed": "iex"}
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            quote = data.get('quote', {})
            ask = quote.get('ap')
            bid = quote.get('bp')
            price = None
            if ask and bid:
                price = (ask + bid) / 2
            elif ask:
                price = ask
            elif bid:
                price = bid

            if price and price > 0:
                print(f"[Server/Alpaca] {ticker}: ${price:.2f}")
                return float(price)
    except Exception as e:
        print(f"[Server/Alpaca] {ticker} Fehler: {e}")
    return None


def get_price_finnhub(ticker: str) -> Optional[float]:
    """Holt Live-Preis von Finnhub (Backup, 60 req/min kostenlos)"""
    api_key = API_KEYS.get('finnhub', '')
    if not api_key:
        return None
    
    try:
        _rate_limit('finnhub')
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            price = data.get('c', 0)
            if price and price > 0:
                print(f"[Server/Finnhub] {ticker}: ${price:.2f}")
                return float(price)
    except Exception as e:
        print(f"[Server/Finnhub] {ticker} Fehler: {e}")
    return None


def get_price_yahoo(ticker: str) -> Optional[float]:
    """Holt Live-Preis von Yahoo Finance (Fallback)"""
    try:
        _rate_limit('yahoo')
        stock = yf.Ticker(ticker)
        
        # Try fast_info first
        try:
            price = stock.fast_info.get('lastPrice')
            if price and price > 0:
                print(f"[Server/Yahoo] {ticker}: ${price:.2f}")
                return float(price)
        except:
            pass
        
        # Fallback to history
        data = stock.history(period='1d', interval='1m')
        if not data.empty:
            price = data['Close'].iloc[-1]
            if price and price > 0:
                print(f"[Server/Yahoo] {ticker}: ${price:.2f}")
                return float(price)
    except Exception as e:
        print(f"[Server/Yahoo] {ticker} Fehler: {e}")
    return None


def get_live_price_single(ticker: str) -> Optional[float]:
    """
    Holt einen einzelnen Live-Preis mit Multi-Provider-Fallback.
    Reihenfolge: Alpaca (Primary) -> Finnhub (Backup) -> Yahoo (Fallback)
    """
    # Provider cascade: fastest/most reliable first
    providers = [
        ('alpaca', get_price_alpaca),      # Primary: Unlimited, fast
        ('finnhub', get_price_finnhub),     # Backup: 60/min limit
        ('yahoo', get_price_yahoo),         # Fallback: Slow but reliable
    ]
    
    for provider_name, provider_func in providers:
        # Skip provider if no API Key (except Yahoo)
        if provider_name != 'yahoo' and not API_KEYS.get(provider_name):
            continue
            
        price = provider_func(ticker)
        if price is not None and price > 0:
            return price
    
    print(f"[Server/WARNING] Kein Preis für {ticker} von allen Providern")
    return None


def get_live_prices(tickers) -> Dict[str, Optional[float]]:
    """
    Fetch live prices for multiple tickers with multi-provider fallback.
    Order: Alpaca (Primary) -> Finnhub (Backup) -> Yahoo (Fallback)
    Uses parallel requests for Alpaca to speed up fetching.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    prices = {}
    
    # Try parallel fetch with Alpaca first (no rate limit)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(get_price_alpaca, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                price = future.result()
                if price is not None and price > 0:
                    prices[ticker] = price
            except Exception as e:
                print(f"[Server/Parallel] {ticker} Error: {e}")
    
    # Fallback for missing prices (sequentially)
    missing = [t for t in tickers if t not in prices or prices[t] is None]
    if missing:
        print(f"[Server] {len(missing)} Ticker brauchen Fallback: {missing[:5]}...")
        for ticker in missing:
            # Try Finnhub then Yahoo
            price = get_price_finnhub(ticker)
            if price is None:
                price = get_price_yahoo(ticker)
            prices[ticker] = price
    
    return prices


class CachedDataFetcher:
    """Cached data fetcher for server-side training"""
    
    def __init__(self, cache_days: int = 1):
        self.cache = {}  # ticker -> (data, timestamp)
        self.cache_days = cache_days
        self.last_error = None
    
    def fetch_historical_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch historical data with caching"""
        try:
            # Check cache
            if ticker in self.cache:
                data, timestamp = self.cache[ticker]
                if datetime.now() - timestamp < timedelta(days=self.cache_days):
                    return data
            
            # Fetch fresh data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            data = fetch_stock_data(ticker, start_date, end_date)
            
            if data is None or data.empty:
                self.last_error = f"No data for {ticker}"
                return None
            
            # Cache the result
            self.cache[ticker] = (data, datetime.now())
            return data
            
        except Exception as e:
            self.last_error = str(e)
            print(f"[CachedDataFetcher] Error fetching {ticker}: {e}")
            return None


def is_market_open():
    """Check if US stock market is currently open (NYSE/NASDAQ hours).
    
    Market hours: 9:30 AM - 4:00 PM Eastern Time (ET)
    Pre-market: 4:00 AM - 9:30 AM ET
    After-hours: 4:00 PM - 8:00 PM ET
    
    We consider 'open' to include pre-market and after-hours for live data.
    """
    try:
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Extended hours: 4:00 AM to 8:00 PM ET
        market_open = now.replace(hour=4, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except ImportError:
        # pytz not available, assume market is open during reasonable hours
        now = datetime.now(timezone.utc)
        # Rough check: UTC 9:00-21:00 covers most of US trading hours
        return 9 <= now.hour <= 21 and now.weekday() < 5
    except Exception:
        # If any error, default to market open (safer for trading)
        return True

def get_live_prices_ib(tickers):
    """Fetch live prices from Interactive Brokers via ib_insync"""
    prices = {}
    try:
        # Allow nested event loops in Streamlit
        import nest_asyncio
        nest_asyncio.apply()
        
        from ib_insync import IB, Stock
        
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)  # Connect to IB Gateway/TWS
        
        for ticker in tickers:
            try:
                contract = Stock(ticker, 'SMART', 'USD')
                ib.qualifyContracts(contract)
                ticker_data = ib.reqMktData(contract, '', False, False)
                ib.sleep(0.1)  # Brief delay to get latest data
                
                if ticker_data and ticker_data.last:
                    prices[ticker] = ticker_data.last
                else:
                    prices[ticker] = None
            except Exception as e:
                print(f"[ERROR] IB-Abruf für {ticker} fehlgeschlagen: {e}")
                prices[ticker] = None
        
        ib.disconnect()
    except Exception as e:
        print(f"IB-Verbindung fehlgeschlagen: {e}. Fallback zu yfinance.")
        return get_live_prices(tickers)
    
    return prices


def fetch_stock_data_alpaca(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical data from Alpaca API (more reliable than yfinance)"""
    api_key = API_KEYS.get('alpaca', '')
    secret_key = API_KEYS.get('alpaca_secret', '')
    base_url = API_KEYS.get('alpaca_base_url', 'https://data.alpaca.markets')
    
    if not api_key or not secret_key:
        return pd.DataFrame()
    
    try:
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        
        # Alpaca uses ISO format dates
        url = f"{base_url}/v2/stocks/{ticker}/bars"
        params = {
            "start": start_date,
            "end": end_date,
            "timeframe": "1Day",
            "feed": "iex",
            "limit": 10000
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            bars = data.get('bars', [])
            
            if not bars:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High', 
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                't': 'Date'
            })
            
            # Keep only required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols]
            
            print(f"[{ticker}/Alpaca] {len(df)} Datenpunkte geladen")
            return df
        else:
            print(f"[{ticker}/Alpaca] HTTP {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[{ticker}/Alpaca] Fehler: {e}")
        return pd.DataFrame()


def fetch_stock_data(ticker, start_date, end_date, max_retries=3):
    """Fetch historical stock data with retry logic and Alpaca fallback"""
    
    # Try Alpaca first (more reliable)
    data = fetch_stock_data_alpaca(ticker, start_date, end_date)
    if not data.empty:
        return data
    
    # Fallback to yfinance with retries
    for attempt in range(max_retries):
        try:
            # Add small delay between retries
            if attempt > 0:
                time.sleep(1 + attempt)
                print(f"[{ticker}] Retry {attempt + 1}/{max_retries}...")
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                if attempt == max_retries - 1:
                    print(f"[{ticker}] Keine Daten nach {max_retries} Versuchen")
                continue
            
            # Reset index to ensure it's not MultiIndex
            data = data.reset_index()
            # Flatten multi-index columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adj Close' in data.columns and 'Close' not in data.columns:
                data['Close'] = data['Adj Close']
            
            # Check all required columns exist
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                print(f"[{ticker}] Fehlende Spalten: {missing}")
                continue
            
            data = data[required_cols]
            print(f"[{ticker}/Yahoo] {len(data)} Datenpunkte geladen")
            return data
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] {ticker} Datenabruf fehlgeschlagen nach {max_retries} Versuchen: {str(e)}")
            continue
    
    return pd.DataFrame()

def fetch_multiple_stocks(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        all_data[ticker] = data
    return all_data

def simulate_live_data(tickers, historical_data, real_prices=None):
    """Simulate live updates by extending historical data with realistic OHLC bars.
    
    Uses REAL current prices from Yahoo Finance as base, then applies small
    intraday volatility. This ensures predictions stay aligned with actual market prices.
    
    Args:
        tickers: List of stock tickers
        historical_data: Dictionary of historical DataFrames per ticker
        real_prices: Optional dict of real current prices (fetched if not provided)
    """
    live_prices = {}
    simulated_extended_data = {}
    
    if not tickers or not historical_data:
        print("[WARNING] Keine Ticker oder historische Daten für Simulation vorhanden")
        return live_prices, simulated_extended_data
    
    # Fetch real prices if not provided (do this once for all tickers)
    if real_prices is None:
        real_prices = get_live_prices(tickers)
    
    for ticker in tickers:
        if ticker in historical_data and historical_data[ticker] is not None and not historical_data[ticker].empty:
            # Use ORIGINAL historical data (not previously extended data)
            original_data = historical_data[ticker].copy()
            
            # Trim to reasonable size
            if len(original_data) > 500:
                original_data = original_data.tail(500)
            
            # Get the REAL current price as base (not old historical data)
            if ticker in real_prices and real_prices[ticker] is not None:
                base_price = real_prices[ticker]
            else:
                # Fallback to last historical close if real price unavailable
                try:
                    fallback_value = original_data['Close'].iloc[-1]
                    if isinstance(fallback_value, (int, float)):
                        base_price = float(fallback_value)
                    else:
                        base_price = float(fallback_value.values[0]) if hasattr(fallback_value, 'values') else float(fallback_value)
                except (ValueError, TypeError, KeyError, IndexError):
                    # Skip ticker if we can't get a price
                    live_prices[ticker] = None
                    continue
            
            # Simulate small intraday movement (-0.3% to +0.3%)
            price_movement = random.uniform(-0.003, 0.003)
            new_close = base_price * (1 + price_movement)
            
            # Simulate Open, High, Low based on the new Close
            new_open = base_price
            new_high = max(new_open, new_close) * (1 + random.uniform(0, 0.001))
            new_low = min(new_open, new_close) * (1 - random.uniform(0, 0.001))
            
            # Scale historical data to match current price level
            # This ensures the LSTM sees data that leads up to current prices
            last_close_value = original_data['Close'].iloc[-1]
            if isinstance(last_close_value, (int, float)):
                hist_last_close = float(last_close_value)
            else:
                hist_last_close = float(last_close_value.values[0]) if hasattr(last_close_value, 'values') else float(last_close_value)
            
            scale_factor = base_price / hist_last_close if hist_last_close > 0 else 1.0
            
            # Ensure no duplicate columns by selecting only OHLCV columns and removing duplicates
            if original_data.columns.duplicated().any():
                # Keep only first occurrence of each column
                original_data = original_data.loc[:, ~original_data.columns.duplicated(keep='first')]
            
            # Scale the data by multiplying each column
            scaled_data = original_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            for col in ['Open', 'High', 'Low', 'Close']:
                scaled_data[col] = scaled_data[col] * scale_factor
            
            # Append ONE simulated bar to the scaled historical data
            # Create new row with all the same columns as scaled_data
            new_row = pd.DataFrame({
                'Open': [new_open],
                'High': [new_high],
                'Low': [new_low],
                'Close': [new_close],
                'Volume': [int(random.uniform(50000000, 100000000))]
            })
            
            # Combine by converting to list of dicts and creating new DataFrame
            data_list = scaled_data.to_dict('records')  # Convert old data to list of dicts
            data_list.append(new_row.to_dict('records')[0])  # Add new row dict
            extended_data = pd.DataFrame(data_list, columns=scaled_data.columns)
            
            simulated_extended_data[ticker] = extended_data
            live_prices[ticker] = new_close
        else:
            live_prices[ticker] = None
    
    # Store extended data in session state for later use in predictions
    return live_prices, simulated_extended_data

def save_extended_data(simulated_extended_data):
    """Speichert die erweiterten Simulationsdaten für kontinuierliches Lernen"""
    import pickle
    import os
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        with open('data/simulated_extended_data.pkl', 'wb') as f:
            pickle.dump(simulated_extended_data, f)
    except Exception as e:
        print(f"[ERROR] Erweiterte Daten konnten nicht gespeichert werden: {e}")

def load_extended_data():
    """Lädt die gespeicherten erweiterten Simulationsdaten"""
    import pickle
    import os
    try:
        filepath = 'data/simulated_extended_data.pkl'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if data is None:
                    return {}
                return data
    except FileNotFoundError:
        print("[INFO] Keine erweiterten Daten gefunden, starte neu")
    except Exception as e:
        print(f"[ERROR] Erweiterte Daten konnten nicht geladen werden: {e}")
    return {}

if __name__ == "__main__":
    data = fetch_stock_data('AAPL', '2020-01-01', '2023-01-01')
    data.to_csv('stock_data.csv')
    print("Daten abgerufen und gespeichert.")