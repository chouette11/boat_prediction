
import torch
import torch.nn as nn
import torch.nn.functional as F

LANE_DIM = 8

class SelfAttnRanker(nn.Module):
    """
    Drop-in replacement for DualHeadRanker with race-level interactions.
    I/O signature:
        forward(ctx: (B, ctx_in), boats: (B,6, boat_in), lane_ids: (B,6)) ->
            st_pred:   (B,6)  (regression)
            rank_pred: (B,6)  (higher=better)
            win_logits:(B,6)  (sigmoid->win prob)
    """
    def __init__(
        self,
        ctx_in: int,
        boat_in: int,
        hidden: int = 128,
        lane_dim: int = LANE_DIM,
        n_heads: int = 4,
        attn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lane_emb = nn.Embedding(6, lane_dim)
        self.ctx_fc   = nn.Linear(ctx_in, hidden)
        self.boat_fc  = nn.Linear(boat_in + lane_dim, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2,
            batch_first=True, dropout=dropout, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=attn_layers)

        # Heads
        self.head_st   = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))
        self.head_rank = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))
        self.head_win  = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))

    def forward(self, ctx, boats, lane_ids):
        """
        ctx:   (B, ctx_in)
        boats: (B, 6, boat_in)
        lane_ids: (B, 6)  # 0..5
        """
        B = boats.size(0)
        lane_e = self.lane_emb(lane_ids)                    # (B,6,lane_dim)
        boat_inp = torch.cat([boats, lane_e], dim=-1)       # (B,6, boat_in+lane_dim)

        ctx_emb = self.ctx_fc(ctx).unsqueeze(1)             # (B,1,h)
        boat_emb = self.boat_fc(boat_inp)                   # (B,6,h)

        # Broadcast ctx to each lane then add & attend
        h0 = torch.tanh(ctx_emb + boat_emb)                 # (B,6,h)

        # Race-level interactions
        h = self.encoder(h0)                                # (B,6,h)

        st_pred    = self.head_st(h).squeeze(-1)            # (B,6)
        rank_pred  = self.head_rank(h).squeeze(-1)          # (B,6)
        win_logits = self.head_win(h).squeeze(-1)           # (B,6)
        return st_pred, rank_pred, win_logits
