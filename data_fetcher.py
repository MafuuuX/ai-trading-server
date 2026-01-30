import yfinance as yf
import pandas as pd
import random
from datetime import datetime, timezone

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


def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"[{ticker}] Keine Daten gefunden")
            return pd.DataFrame()
        
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
            print(f"[{ticker}] Fehlende Spalten nach Download: {missing}")
            return pd.DataFrame()
        
        data = data[required_cols]
        print(f"[{ticker}] {len(data)} Datenpunkte geladen")
        return data
    except Exception as e:
        print(f"[ERROR] {ticker} Datenabruf fehlgeschlagen: {str(e)}")
        return pd.DataFrame()

def fetch_multiple_stocks(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        all_data[ticker] = data
    return all_data
def get_live_prices(tickers):
    """Fetch live prices from Yahoo Finance"""
    prices = {}
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period='1d', interval='1m')
            if not data.empty:
                price = data['Close'].iloc[-1]
                if price and price > 0:
                    prices[ticker] = float(price)
                else:
                    print(f"[WARNING] Ungültiger Preis für {ticker}: {price}")
                    prices[ticker] = None
            else:
                print(f"[WARNING] Keine Live-Daten für {ticker}")
                prices[ticker] = None
        except Exception as e:
            print(f"[ERROR] Live-Preis für {ticker} konnte nicht abgerufen werden: {e}")
            prices[ticker] = None
    return prices

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