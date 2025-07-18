#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:







# In[35]:


import torch, tqdm
import pandas as pd, psycopg2, os
from sklearn.preprocessing import StandardScaler
import numpy as np  
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn
import datetime as dt
from dotenv import load_dotenv


# In[36]:


load_dotenv()

DB_CONF = {
    "host":     os.getenv("PGHOST", "localhost"),
    "port":     int(os.getenv("PGPORT", 5432)),
    "dbname":   os.getenv("PGDATABASE", "boatrace"),
    "user":     os.getenv("PGUSER", "br_user"),
    "password": os.getenv("PGPASSWORD", "secret"),
}

# ------------------------------------------------------------------
# DB 接続
# ------------------------------------------------------------------
conn = psycopg2.connect(**DB_CONF)
df = pd.read_sql("""
    SELECT * FROM feat.train_features
    WHERE race_date <= '2024-12-31'
""", conn)

print(f"Loaded {len(df)} rows from the database.")


# In[37]:


NUM_COLS = ["air_temp", "wind_speed", "wave_height", "water_temp"]
scaler = StandardScaler().fit(df[NUM_COLS])
df[NUM_COLS] = scaler.transform(df[NUM_COLS])

bool_cols = [c for c in df.columns if c.endswith("_fs_flag")]
df[bool_cols] = df[bool_cols].fillna(False)

rank_cols = [f"lane{l}_rank" for l in range(1, 7)]
df[rank_cols] = df[rank_cols].fillna(7).astype("int32")
df.to_csv("artifacts/train_features.csv", index=False)
display(df.head())
print("データフレーム全体の欠損値の総数:", df.isnull().sum().sum())

# 各列の欠損値の割合を表示（0〜1の値）
missing_ratio = df.isnull().mean()

# パーセント表示にする場合（見やすさのため）
missing_ratio_percent = missing_ratio * 100

print("各列の欠損値の割合（%）:")
print(missing_ratio_percent.sort_values(ascending=False))

os.makedirs("artifacts", exist_ok=True)
scaler_filename = "artifacts/wind_scaler.pkl"
joblib.dump(scaler, scaler_filename)


# In[38]:


def encode(col):
    uniq = sorted(df[col].dropna().unique())
    mapping = {v:i for i,v in enumerate(uniq)}
    df[col + "_id"] = df[col].map(mapping).fillna(-1).astype("int16")
    return mapping
venue2id = encode("venue")
# race_type2id = encode("race_type")


# In[39]:


class BoatRaceDataset(Dataset):
    """
    - 数値列: float32, NaN/±inf → 0.0
    - rank ∈ {1,2,3,4,5,6,…}  (重複可) を
      *重複しない 1〜6 & 最下位以降* に正規化して返す
    """
    def __init__(self, frame: pd.DataFrame, mode: str = "diff"):
        self.f = frame.copy()
        self.mode = mode

        # --- 数値列を float32, 欠損→0.0 -------------------------------
        num_cols = self.f.select_dtypes(include=["number", "bool"]).columns
        self.f[num_cols] = (
            self.f[num_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )

        if mode == "zscore":
            self.boat_scaler = StandardScaler()
            boat_feats = []
            for lane in range(1, 7):
                boat_feats.append(self.f[[f"lane{lane}_exh_time", f"lane{lane}_st", f"lane{lane}_weight"]].values)
            boat_all = np.stack(boat_feats, axis=1).reshape(-1, 3)  # shape (N*6, 3)
            self.boat_scaler.fit(boat_all)

        # --- rank を int64 で保存 (欠損→99) ---------------------------
        for lane in range(1, 7):
            col = f"lane{lane}_rank"
            if col in self.f.columns:
                self.f[col] = (
                    self.f[col]
                    .fillna(99)          # 欠損は論外扱い
                    .astype("int64")
                )

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        r = self.f.iloc[idx]

        # ❶ 環境特徴量 --------------------------------------------------
        ctx = torch.tensor([
            r["wind_speed"], r["wave_height"],
            r["air_temp"],   r["water_temp"],
        ], dtype=torch.float32)

        # ❷ 各艇の元特徴量を収集 ---------------------------------------
        exh_times = [r[f"lane{lane}_exh_time"] for lane in range(1, 7)]
        st_times  = [r[f"lane{lane}_st"] for lane in range(1, 7)]
        fs_flags  = [float(r[f"lane{lane}_fs_flag"]) for lane in range(1, 7)]
        weights   = [r[f"lane{lane}_weight"] for lane in range(1, 7)]
        raw_ranks = [int(r[f"lane{lane}_rank"]) for lane in range(1, 7)]
        lane_ids  = list(range(6))

        boats = []
        for i in range(6):
            if self.mode == "diff":
                mean_exh = np.mean(exh_times)
                mean_st  = np.mean(st_times)
                mean_wt  = np.mean(weights)
                feat = [
                    exh_times[i] - mean_exh,
                    st_times[i]  - mean_st,
                    fs_flags[i],
                    weights[i]   - mean_wt,
                ]
            elif self.mode == "raw":
                feat = [
                    exh_times[i],
                    st_times[i],
                    fs_flags[i],
                    weights[i],
                ]
            elif self.mode == "log":
                feat = [
                    np.log1p(exh_times[i]),
                    np.log1p(st_times[i]),
                    fs_flags[i],
                    np.log1p(weights[i]),
                ]
            elif self.mode == "zscore":
                inp = np.array([[exh_times[i], st_times[i], weights[i]]])
                scaled = self.boat_scaler.transform(inp)[0]
                feat = [
                    scaled[0],
                    scaled[1],
                    fs_flags[i],
                    scaled[2],
                ]
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            boats.append(torch.tensor(feat, dtype=torch.float32))

        # ---------- ★ 重複しない順位を付け直す ★ ----------------------
        # 例: [1, 2, 6, 3, 6, 6] → [1, 2, 4, 3, 5, 6]
        order = np.argsort(raw_ranks)          # 小さい順に艇 index を並べる
        new_rank = [0]*6
        for new_pos, lane_idx in enumerate(order, start=1):  # new_pos:1..6
            new_rank[lane_idx] = new_pos       # 一意な 1..6 を付け直し

        return (
            ctx,
            torch.stack(boats),
            torch.tensor(lane_ids, dtype=torch.int64),
            torch.tensor(new_rank, dtype=torch.int64)
        )


# In[40]:


# ============================================================
# 0) ── データの“ラベル & 特徴量”を 1 行だけ覗く可視化 Snippet
#      ★★ ここは notebook なら「1 セルだけ」実行すれば OK ★★
# ------------------------------------------------------------
def peek_one(df: pd.DataFrame, seed: int = 0) -> None:
    """
    ランダムに 1 レース（1 行）だけ抜き取り、順位と主要特徴量を一覧表示
    """
    row = df.sample(1, random_state=seed).squeeze()

    def lane_list(prefix: str):
        return [row[f"lane{i}_{prefix}"] for i in range(1, 7)]

    print("── sample race ──")
    print("rank    :", lane_list("rank"))
    print("exh_time:", lane_list("exh_time"))
    print("st      :", lane_list("st"))
    print("fs_flag :", lane_list("fs_flag"))
    print("weight  :", lane_list("weight"))

# ---------------------------------------------
# ここで一度だけ呼んで目視確認しておくとズレにすぐ気付けます
peek_one(df)
# ============================================================


LANE_DIM = 8
class SimpleCPLNet(nn.Module):
    """
    ctx(4) + boat(4) → lane ごとにスコア 1 個
    """
    def __init__(self, ctx_in=4, boat_in=4, hidden=64, lane_dim=LANE_DIM):
        super().__init__()
        self.lane_emb = nn.Embedding(6, lane_dim)
        self.ctx_fc   = nn.Linear(ctx_in, hidden)
        self.boat_fc  = nn.Linear(boat_in + lane_dim, hidden)
        self.head     = nn.Linear(hidden, 1)

        # 重み初期化を対称性ブレイク用に Xavier で揃える
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, ctx, boats, lane_ids):  # boats:(B,6,4) lane_ids:(B,6)
        B, L, _ = boats.size()
        ctx_emb  = self.ctx_fc(ctx)           # (B,h)
        # DataLoader から来る lane_ids が (B,) なら (B,6) へブロードキャスト
        # -------- lane_ids の形状を必ず (B,6) にそろえる --------
        if lane_ids.dim() == 1:               # (B,) → (B,6)
            lane_ids = lane_ids.unsqueeze(1).expand(-1, L)
        elif lane_ids.dim() == 2 and lane_ids.size(1) == 1:  # (B,1) → (B,6)
            lane_ids = lane_ids.expand(-1, L)
        # 以外 (既に (B,6)) はそのままで OK
        lane_ids = lane_ids.contiguous()      # Embedding 要求に備え contiguous 化


        lane_emb = self.lane_emb(lane_ids)    # (B,6,lane_dim)
        boat_inp = torch.cat([boats, lane_emb], dim=-1)
        boat_emb = self.boat_fc(boat_inp)     # (B,6,h)

        # broadcast ctx → 各 lane
        score = self.head(torch.tanh(ctx_emb.unsqueeze(1) + boat_emb))  # (B,6,1)
        return score.squeeze(-1)           # (B,6)


# In[41]:


def pl_nll(scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """
    scores : (B, 6) ― lane0 – lane5 のスコア
    ranks  : (B, 6) ― **1 が 1 着, … 6 が 6 着**（列番号ではない）
   """
    scores = scores.clamp(-20.0, 20.0)

    # 着順 (1 → 6) に並んだ lane index を取得
    order = torch.argsort(ranks, dim=1)      # shape (B,6)

    nll = torch.zeros(scores.size(0), device=scores.device)
    s   = scores.clone()
    for pos in range(6):
        log_denom = torch.logsumexp(s, dim=1)            # log Σₗ exp
        idx       = order[:, pos]                        # (B,)
        chosen    = s.gather(1, idx.unsqueeze(1)).squeeze(1)
        nll      += log_denom - chosen
        s         = s.scatter(1, idx.unsqueeze(1), float('-inf'))

    return nll.mean()

# ── pl_nll が正しいか 3 秒で判定 ──
scores = torch.tensor([[6, 5, 4, 3, 2, 1]], dtype=torch.float32)  # lane0 が最強
ranks  = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)    # lane0 が 1 着
print("pl_nll should be ~0 :", pl_nll(scores, ranks).item())


# In[42]:


df["race_date"] = pd.to_datetime(df["race_date"]).dt.date
cutoff = dt.date(2024, 11, 1)

ds_train = BoatRaceDataset(df[df["race_date"] <  cutoff])
ds_val   = BoatRaceDataset(df[df["race_date"] >= cutoff])
print(f"train: {len(ds_train)}  val: {len(ds_val)}")
# print("train:", ds_train[0])  # 1 レースの特徴量を確認

loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
loader_val   = DataLoader(ds_val,   batch_size=512)

# ------------------- ⑤ 学習ループ（LR↓ + Clip） --------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SimpleCPLNet().to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

EPOCHS = 30
for epoch in range(EPOCHS):

    if epoch == 0:                  # 1 エポック目だけ試す例
        ctx, boats, lane_ids, ranks = next(iter(loader_train))
        ctx, boats = ctx.to(device), boats.to(device)
        lane_ids = lane_ids.to(device)

        scores = model(ctx, boats, lane_ids)
        scores.sum().backward()     # ダミー backward
        grad_norm = sum(p.grad.abs().mean().item() for p in model.parameters())
        print(f"[debug] average |grad| = {grad_norm:.3e}")
    # ---- train ----
    model.train(); tr_sum = 0
    for ctx, boats, lane_ids, ranks in loader_train:
        ctx, boats = ctx.to(device), boats.to(device)
        lane_ids, ranks = lane_ids.to(device), ranks.to(device)

        loss = pl_nll(model(ctx, boats, lane_ids), ranks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # ★勾配爆発対策★
        opt.step()

        tr_sum += loss.item() * len(ctx)

    tr_nll = tr_sum / len(loader_train.dataset)

    # ---- validation ----
    model.eval(); val_sum = 0
    with torch.no_grad():
        for ctx, boats, lane_ids, ranks in loader_val:
            ctx, boats = ctx.to(device), boats.to(device)
            lane_ids, ranks = lane_ids.to(device), ranks.to(device)
            val_sum += pl_nll(model(ctx, boats, lane_ids), ranks).item() * len(ctx)

    val_nll = val_sum / len(loader_val.dataset)

    print(f"epoch {epoch:2d}  train_nll {tr_nll:.4f}  val_nll {val_nll:.4f}")

    # ---- accuracy & 三連単的中率 ----
    def top1_accuracy(scores, ranks):
        pred_top1 = scores.argmax(dim=1)
        true_top1 = (ranks == 1).nonzero(as_tuple=True)[1]
        return (pred_top1 == true_top1).float().mean().item()

    def trifecta_hit_rate(scores, ranks):
        """
        予測スコア上位3着までと、実際の着順上位3着の組み合わせ一致を見る（順不同）
        """
        pred_top3 = torch.topk(scores, k=3, dim=1).indices
        true_top3 = torch.topk(-ranks, k=3, dim=1).indices  # 小さい順に上位3着
        hit = [set(p.tolist()) == set(t.tolist()) for p, t in zip(pred_top3, true_top3)]
        return sum(hit) / len(hit)

    # accuracy 評価
    model.eval(); all_scores, all_ranks = [], []
    with torch.no_grad():
        for ctx, boats, lane_ids, ranks in loader_val:
            ctx, boats = ctx.to(device), boats.to(device)
            lane_ids = lane_ids.to(device)
            scores = model(ctx, boats, lane_ids).cpu()
            all_scores.append(scores)
            all_ranks.append(ranks)

    all_scores = torch.cat(all_scores, dim=0)
    all_ranks = torch.cat(all_ranks, dim=0)

    acc_top1 = top1_accuracy(all_scores, all_ranks)
    acc_tri3 = trifecta_hit_rate(all_scores, all_ranks)

    print(f"Top-1 Acc: {acc_top1:.3f}   Trifecta Hit: {acc_tri3:.3f}")


# In[ ]:


# ============================================================
# ④ ── 「勾配が流れているか」を瞬時に確認する Snippet
#       （エポック終了後 1 回だけ走らせれば十分）
# ------------------------------------------------------------

# ============================================================
 
 # ============================================================
 # ⑤ ── 超小規模データで「過学習できるか」テスト関数
 #       必要時に呼び出して 0.1 以下まで loss が落ちるか確認
 # ------------------------------------------------------------
def overfit_tiny(df: pd.DataFrame, device: str = "cpu"):
    """
    データセットを 10 行だけに縮小し、500 step で過学習できるか検証
    """
    tiny_df = df.sample(10, random_state=1).reset_index(drop=True)
    tiny_ds = BoatRaceDataset(tiny_df)
    tiny_loader = DataLoader(tiny_ds, batch_size=10, shuffle=True)

    net = SimpleCPLNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)

    for _ in range(500):
        ctx, boats, lane_ids, ranks = next(iter(tiny_loader))
        ctx, boats = ctx.to(device), boats.to(device)
        lane_ids, ranks = lane_ids.to(device), ranks.to(device)

        loss = pl_nll(net(ctx, boats, lane_ids), ranks)
        opt.zero_grad(); loss.backward(); opt.step()

    print("[tiny] final loss:", loss.item())


# ---- tiny データで特徴量の分散を確認 -----------------------
tiny_df = df.sample(10, random_state=1).reset_index(drop=True)
num_cols = tiny_df.select_dtypes(include="number").columns

# (1) 行間（=レース間）での分散
print("► 行間 variance (should be >0):")
print(tiny_df[num_cols].var().nsmallest(10))

# (2) 同一レース内（= 6 艇間）での分散
def per_race_var(col):
    return tiny_df.groupby("race_key")[col].var().mean()

per_race = {c: per_race_var(c) for c in num_cols}
print("\n► 6 艇間 variance:")
print(sorted(per_race.items(), key=lambda x: x[1])[:10])

# ---- 呼び方例 ----
overfit_tiny(df, device)
# ============================================================


# In[ ]:


torch.save({
    "state_dict": model.state_dict(),
    "scaler": scaler_filename,
    "venue2id": venue2id,
    # "race_type2id": race_type2id
}, "cplnet_checkpoint.pt")


# In[ ]:


# .ipynbを.pyに変換しておく
if __name__ == "__main__":
    import nbformat
    from nbconvert import PythonExporter

    with open("main.ipynb", "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)

    with open("main.py", "w", encoding="utf-8") as f:
        f.write(source)

