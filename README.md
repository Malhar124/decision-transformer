# Financial Decision Transformer: Algorithmic Trading via Offline RL

This repository implements a **Decision Transformer (DT)** for algorithmic trading. By treating Reinforcement Learning (RL) as a causal sequence modeling problem, this architecture bypasses the instability of traditional Temporal Difference (TD) learning. Instead of exploring a live environment, the model is trained offline on historical market data and can be "prompted" with a target financial return to autoregressively generate the optimal sequence of trades zero-shot.

## Paradigm Shift: Trading as Sequence Modeling

Traditional RL in finance struggles with noisy environments and fragile value functions. This project leverages the **Offline RL** paradigm introduced by Chen et al. (2021).

The transformer takes a sequence of triplets:
`[Return-to-Go, State, Action] ... [Return-to-Go, State, Action]`

By training a causal GPT-style architecture to predict the next `Action` based on the historical context and the desired `Return-to-Go` (cumulative future profit), we can literally prompt the neural network during live markets: *"Given this recent price action, what trade should I execute right now to achieve a +15% return over the next quarter?"*

## Architecture & Data Pipeline

### 1. The Offline Dataset (`yfinance`)

* **State ($s_t$):** OHLCV data merged with standard technical indicators (SMA_10, SMA_50, Rolling Volatility). All states are Z-score normalized.
* **Action ($a_t$):** Discrete action space: `0` (Sell/Short), `1` (Hold), `2` (Buy/Long). Expert trajectories are generated using deterministic moving-average crossovers.
* **Return-to-Go ($\hat{R}_t$):** The cumulative future return over an episodic window of 60 trading days.

### 2. The Sequence Model

* **Embeddings:** Continuous modalities (State, RTG) are projected via linear layers; Actions use discrete embeddings. All are fused into a hidden dimension of `d=128`.
* **Transformer:** 3-layer Causal Transformer Encoder (`nhead=4`). A strict upper-triangular causal mask prevents look-ahead bias.
* **Context Window:** $K = 20$. The model analyzes the past 20 trading days to predict the optimal action for the current step.

## Training Results

The model was trained using standard Cross-Entropy Loss exclusively on the Action prediction head. Because the environment is offline and static, training is exceptionally stable and fits easily within a standard 8GB VRAM GPU.

* **Initial Action Loss:** `0.1222`
* **Final Action Loss (Epoch 30):** `0.0365`
* **Convergence:** Smooth autoregressive alignment with the expert dataset without collapsing into local minima.

## Live Inference & Execution Bridge

The model transitions from a PyTorch sequence predictor to a live execution engine by generating JSON payloads ready for broker webhooks (e.g., MetaTrader5, Alpaca, Dhan).

**Example: Prompting for a +15% Target Return**

```python
# Feed the last 20 days of market data and demand a 15% return
live_rtgs = torch.full((1, CONTEXT_LEN, 1), 0.15) 
action_logits = model(live_states, live_actions, live_rtgs)

```

**Generated Webhook Payload:**

```json
{
    "symbol": "SPY",
    "action": "BUY",
    "confidence": 0.9999135732650757,
    "prompted_target_return": 0.15,
    "order_type": "MARKET",
    "timestamp": "2026-05-23T15:56:31.914831"
}

```

## 🛠️ Repository Structure

```text
├── data/
│   ├── dataset_generation.py         # Pulls OHLCV and calculates states/RTG
├── models/
│   ├── decision_transformer.py       # PyTorch causal transformer architecture
├── scripts/
│   ├── train_offline_rl.py           # Dataloader and training loop
│   ├── live_inference.py             # Broker webhook generation script
├── requirements.txt
└── README.md

```

## Quick Start

**1. Install Dependencies**

```bash
pip install -r requirements.txt

```

**2. Generate Offline Data & Train**

```bash
python data/dataset_generation.py --ticker SPY --start 2020-01-01 --end 2023-01-01
python scripts/train_offline_rl.py --epochs 30 --context_len 20

```

**3. Run Live Inference Prompting**

```bash
python scripts/live_inference.py --target_return 0.15

```

## Disclaimer

This repository is for **academic and research purposes only**. The models and algorithmic strategies provided do not constitute financial advice. Algorithmic trading carries significant financial risk. Never deploy an experimental machine learning model to a live brokerage account without extensive paper trading, risk management protocols, and capital you can afford to lose.

*** **Author:** Malhar Udmale

**Institution:** Indian Institute of Information Technology, Allahabad

**Contact:** malharudmale@gmail.com | [LinkedIn](www.linkedin.com/in/malhar-udmale-a83772324)
