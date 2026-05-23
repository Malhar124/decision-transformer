import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import os

def generate_dataset(ticker, start_date, end_date, output_path):
    print(f"Downloading historical market data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}. Check dates or ticker symbol.")

    # 1. State Engineering
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    df = df.dropna()

    # Normalize states (Z-score)
    # Note: In production, you would save these means/stds to scale live data.
    state_cols = ['Return', 'SMA_10', 'SMA_50', 'Volatility']
    for col in state_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # 2. Expert Action Generation (0: Short, 1: Hold, 2: Long)
    conditions = [
        (df['SMA_10'] > df['SMA_50']), # Bullish crossover
        (df['SMA_10'] < df['SMA_50'])  # Bearish crossover
    ]
    choices = [2, 0]
    df['Action'] = np.select(conditions, choices, default=1)

    # 3. Return-to-Go (RTG) Calculation
    EPISODE_LENGTH = 60
    rtg_list = []
    for i in range(len(df)):
        end_idx = min(i + EPISODE_LENGTH, len(df))
        future_returns = df['Return'].iloc[i:end_idx].sum() 
        rtg_list.append(future_returns)

    df['RTG'] = rtg_list

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path)
    print(f"Successfully generated Offline Trading Dataset: {len(df)} timesteps.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Offline RL Dataset")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data/offline_dataset.csv", help="Output CSV path")
    
    args = parser.parse_args()
    generate_dataset(args.ticker, args.start, args.end, args.output)