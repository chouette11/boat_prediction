import torch.nn as nn
import torch


LANE_DIM = 8

class DualHeadRanker(nn.Module):
    """
    ctx(6) + boat(6) → lane ごとにスコア 1 個
    """
    def __init__(self, ctx_in=6, boat_in=6, hidden=160, lane_dim=LANE_DIM):
        super().__init__()
        self.lane_emb = nn.Embedding(6, lane_dim)
        self.ctx_fc   = nn.Linear(ctx_in, hidden)
        self.boat_fc  = nn.Linear(boat_in + lane_dim, hidden)
        self.head_rank = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.head_st = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.head_win = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # 重み初期化を対称性ブレイク用に Xavier で揃える
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, ctx, boats, lane_ids):  # boats:(B,6,4) lane_ids:(B,6)
        B, L, _ = boats.size()
        ctx_emb  = self.ctx_fc(ctx)          # (B,h)

        if lane_ids.dim() == 1:
            lane_ids = lane_ids.unsqueeze(1).expand(-1, L)
        elif lane_ids.dim() == 2 and lane_ids.size(1) == 1:
            lane_ids = lane_ids.expand(-1, L)
        lane_ids = lane_ids.contiguous()

        lane_emb = self.lane_emb(lane_ids)   # (B,6,lane_dim)
        boat_inp = torch.cat([boats, lane_emb], dim=-1)
        boat_emb = self.boat_fc(boat_inp)    # (B,6,h)

        h = torch.tanh(ctx_emb.unsqueeze(1) + boat_emb)  # (B,6,h)

        st_pred   = self.head_st(h).squeeze(-1)     # (B,6)
        rank_pred = self.head_rank(h).squeeze(-1)   # (B,6)
        win_logits = self.head_win(h).squeeze(-1)   # (B,6)
        return st_pred, rank_pred, win_logits

# --- alias for legacy references ---
SimpleCPLNet = DualHeadRanker