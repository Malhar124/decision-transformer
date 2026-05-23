import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import sys

# Ensure models module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.decision_transformer import TradingDecisionTransformer

class TradingTrajectoryDataset(Dataset):
    def __init__(self, df, context_len=20):
        self.context_len = context_len
        state_cols = ['Return', 'SMA_10', 'SMA_50', 'Volatility']
        self.states = df[state_cols].values.astype(np.float32)
        self.actions = df['Action'].values.astype(np.int64)
        self.rtgs = df['RTG'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.states) - self.context_len
        
    def __getitem__(self, idx):
        s = self.states[idx : idx + self.context_len]
        a = self.actions[idx : idx + self.context_len]
        r = self.rtgs[idx : idx + self.context_len]
        return torch.tensor(s), torch.tensor(a), torch.tensor(r).unsqueeze(-1)

def train(epochs, context_len, data_path, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Run dataset_generation.py first.")
    
    df = pd.read_csv(data_path)
    dataset = TradingTrajectoryDataset(df, context_len=context_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize Model
    state_dim = 4 # Return, SMA_10, SMA_50, Volatility
    model = TradingDecisionTransformer(state_dim=state_dim, action_dim=3, max_length=context_len).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Starting Offline RL Training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for states, actions, rtgs in dataloader:
            states, actions, rtgs = states.to(device), actions.to(device), rtgs.to(device)
            
            optimizer.zero_grad()
            action_preds = model(states, actions, rtgs)
            
            # Shift predictions to align with the *next* action target
            preds = action_preds[:, :-1, :].reshape(-1, 3)
            targets = actions[:, 1:].reshape(-1)
            
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Action Cross-Entropy Loss: {avg_loss:.4f}")

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Training Complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--data_path", type=str, default="data/offline_dataset.csv")
    parser.add_argument("--model_save", type=str, default="models/dt_model.pth")
    
    args = parser.parse_args()
    train(args.epochs, args.context_len, args.data_path, args.model_save)