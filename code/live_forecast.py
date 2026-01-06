import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from xgboost import XGBRegressor
from datetime import datetime

# Reuse your existing configuration classes or simple config
class Config:
    mode = "DJI"  # Change to "ATHEX" if needed
    horizon_days = 126
    top_n = 5
    cache_dir = "market_data_cache"
    
    # Updated to NOW
    start_date = "2010-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d') 

    if mode == "DJI":
        tickers = [
            "AAPL", "MSFT", "JPM", "V", "PG", "JNJ", "WMT", "KO", 
            "DIS", "MCD", "GS", "IBM", "CAT", "MMM", "AXP", "BA", 
            "CSCO", "CVX", "DOW", "HD", "HON", "INTC", "MRK", 
            "NKE", "TRV", "UNH", "VZ", "CRM", "AMGN", "WBA"
        ]
        macro_map = {"SP500": "^GSPC", "VIX": "^VIX", "Oil": "BZ=F", "Gold": "GC=F", "US10Y": "^TNX"}
    else:
        tickers = [
            "ALPHA.AT", "ETE.AT", "EUROB.AT", "PPC.AT", "OPAP.AT", 
            "MYTIL.AT", "TPEIR.AT", "GEKTERNA.AT", "MOH.AT", "TITC.AT"
        ]
        macro_map = {"EuroStoxx": "^STOXX50E", "VIX": "^VIX"}

# ---------------------------------------------------------
# 1. ROBUST DATA LOADER (Anti-Ban)
# ---------------------------------------------------------
def download_data(cfg):
    print(f"Downloading Live Data for {len(cfg.tickers)} tickers...")
    
    # Custom Session to bypass Yahoo blocks
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    # 1. Tickers
    try:
        df = yf.download(cfg.tickers, start=cfg.start_date, end=cfg.end_date, 
                         group_by='ticker', auto_adjust=False, progress=True, session=session)
    except Exception as e:
        print(f"Download failed: {e}")
        return None, None

    # Extract Adj Close
    adj_close = pd.DataFrame()
    volume = pd.DataFrame()
    
    # Handle MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # If columns are (Ticker, Field) or (Field, Ticker)
        # We try to extract 'Adj Close' for each ticker
        for t in cfg.tickers:
            try:
                # Try accessing ticker first
                if t in df.columns.levels[0]:
                    adj_close[t] = df[t]['Adj Close']
                    volume[t] = df[t]['Volume']
                # Try accessing field first
                elif 'Adj Close' in df.columns.levels[0]:
                    adj_close[t] = df['Adj Close'][t]
                    volume[t] = df['Volume'][t]
            except:
                pass
    else:
        # Single ticker case
        adj_close[cfg.tickers[0]] = df['Adj Close']
        volume[cfg.tickers[0]] = df['Volume']

    # 2. Macro
    print("Downloading Macro...")
    macro_df = yf.download(list(cfg.macro_map.values()), start=cfg.start_date, end=cfg.end_date, 
                           progress=False, auto_adjust=False, session=session)['Adj Close']
    macro_df.columns = [k for k,v in cfg.macro_map.items() if v in macro_df.columns]
    
    return adj_close, macro_df.ffill()

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (PRESERVE LIVE ROWS)
# ---------------------------------------------------------
def build_features(prices, macro, cfg):
    print("Building features...")
    frames = []
    
    # Align macro
    macro = macro.reindex(prices.index).ffill()
    
    for t in prices.columns:
        p = prices[t].dropna()
        if len(p) < 252: continue
        
        df = pd.DataFrame(index=p.index)
        df['ticker'] = t
        df['price'] = p
        
        # Momentum
        df['ret_1m'] = p.pct_change(21)
        df['ret_3m'] = p.pct_change(63)
        df['ret_6m'] = p.pct_change(126)
        
        # Volatility
        df['vol_3m'] = p.pct_change().rolling(63).std()
        
        # RSI
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Target (Shifted)
        # This will be NaN for the last 6 months. WE KEEP IT.
        df['target'] = p.shift(-cfg.horizon_days) / p - 1
        
        # Add Macro
        df = df.join(macro, how='left')
        
        frames.append(df)
        
    return pd.concat(frames)

# ---------------------------------------------------------
# 3. TRAIN & PREDICT
# ---------------------------------------------------------
def generate_signal():
    cfg = Config()
    prices, macro = download_data(cfg)
    
    if prices is None or prices.empty:
        print("No data found!")
        return

    # Build Panel
    panel = build_features(prices, macro, cfg)
    
    # Define Features (exclude target/meta)
    features = [c for c in panel.columns if c not in ['ticker', 'price', 'target', 'prediction']]
    # Drop rows where FEATURES are missing (e.g. start of history)
    panel = panel.dropna(subset=features)
    
    # SPLIT: History vs Live
    # History = We know the target (prices 6 months later exist)
    # Live = Target is NaN (the future hasn't happened)
    
    history = panel.dropna(subset=['target'])
    live = panel[panel['target'].isna()]
    
    # Get the VERY LATEST row for each ticker
    # This represents "Today" (or the last trading day)
    today_data = live.groupby('ticker').tail(1).sort_index()
    latest_date = today_data.index.max()
    
    print(f"\nTraining on {len(history)} historical rows...")
    print(f"Predicting for Date: {latest_date.date()}")
    
    # Train Model
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, n_jobs=-1)
    model.fit(history[features], history['target'])
    
    # Predict
    preds = model.predict(today_data[features])
    today_data['predicted_return'] = preds
    
    # ---------------------------------------------------------
    # 4. OUTPUT RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print(f"🚀 LIVE BUY SIGNALS ({cfg.mode})")
    print(f"Horizon: 6 Months")
    print("="*40)
    
    results = today_data[['ticker', 'price', 'predicted_return']].sort_values('predicted_return', ascending=False)
    
    # Formatting
    results['predicted_return'] = (results['predicted_return'] * 100).map('{:,.1f}%'.format)
    results['price'] = results['price'].map('${:,.2f}'.format)
    
    print(results.head(10))
    
    # Recommendation
    print("\n" + "-"*40)
    print("💡 EXECUTION STRATEGY:")
    print(f"1. Log into Interactive Brokers.")
    print(f"2. Buy the Top 4 stocks: {results['ticker'].iloc[:4].tolist()}")
    print(f"3. Use 'Fractional Shares' if price is too high.")
    print(f"4. Hold for 1 month, then re-run this script.")
    print("-"*40)

if __name__ == "__main__":
    generate_signal()