import torch
import torch.nn as nn

class TradingDecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_length=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Modality Embeddings
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Embedding(action_dim, hidden_dim)
        self.embed_rtg = nn.Linear(1, hidden_dim)
        
        # Time Step Embedding (Positional)
        self.embed_timestep = nn.Embedding(max_length, hidden_dim)
        
        # Causal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim*4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Action Prediction Head
        self.predict_action = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, states, actions, rtgs):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Embed modalities
        state_emb = self.embed_state(states)
        action_emb = self.embed_action(actions)
        rtg_emb = self.embed_rtg(rtgs)
        
        # Add time embeddings
        time_steps = torch.arange(seq_length, device=states.device).unsqueeze(0).repeat(batch_size, 1)
        time_emb = self.embed_timestep(time_steps)
        
        # Integrate embeddings
        token_embeddings = state_emb + action_emb + rtg_emb + time_emb
        
        # Create Causal Mask (prevent looking into the future)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(states.device)
        
        # Pass through Transformer
        x = self.transformer(token_embeddings, is_causal=True, mask=causal_mask)
        
        # Predict the next action logits
        action_preds = self.predict_action(x)
        return action_preds