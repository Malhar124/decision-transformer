import torch
import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import json
import os
import sys

# Ensure models module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.decision_transformer import TradingDecisionTransformer

def get_live_market_states(ticker, context_len):
    # Fetch enough recent data to calculate the 50-day SMA accurately
    df = yf.download(ticker, period="100d", progress=False)
    
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    df = df.dropna()
    
    # Normalize (Using recent historical window for simulation)
    state_cols = ['Return', 'SMA_10', 'SMA_50', 'Volatility']
    for col in state_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
        
    # Extract just the exact context window required
    recent_states = df[state_cols].values[-context_len:]
    return recent_states

def run_inference(ticker, target_return, context_len, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Train the model first.")

    # Initialize model and load weights
    model = TradingDecisionTransformer(state_dim=4, action_dim=3, max_length=context_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 1. Fetch live market state
    recent_states = get_live_market_states(ticker, context_len)
    live_states = torch.tensor(recent_states).unsqueeze(0).to(device, dtype=torch.float32)
    
    # 2. Setup Context Arrays
    # Dummy initialized previous actions for the context window
    live_actions = torch.ones((1, context_len), dtype=torch.int64).to(device) 
    
    # --- THE PROMPT ---
    live_rtgs = torch.full((1, context_len, 1), target_return).to(device, dtype=torch.float32)

    with torch.no_grad():
        # Forward pass
        action_logits = model(live_states, live_actions, live_rtgs)
        
        # Get prediction for the absolute latest timestep
        latest_logits = action_logits[0, -1, :]
        action_idx = torch.argmax(latest_logits).item()

    # Format output
    action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    decision = action_map[action_idx]
    confidence = float(torch.softmax(latest_logits, dim=-1).max())

    webhook_payload = {
        "symbol": ticker,
        "action": decision,
        "confidence": confidence,
        "prompted_target_return": target_return,
        "order_type": "MARKET",
        "timestamp": pd.Timestamp.now().isoformat()
    }

    print("\n--- DECISION TRANSFORMER LIVE INFERENCE ---")
    print(f"Prompted Target: {target_return * 100}% Return")
    print("Generated Broker Payload:")
    print(json.dumps(webhook_payload, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Inference Payload Generator")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--target_return", type=float, default=0.15, help="Target return as a decimal (e.g., 0.15 for 15%%)")
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--model_path", type=str, default="models/dt_model.pth")
    
    args = parser.parse_args()
    run_inference(args.ticker, args.target_return, args.context_len, args.model_path)