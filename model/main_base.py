#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pandas as pd, psycopg2, os
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import numpy as np  
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn
import datetime as dt
from dotenv import load_dotenv
import matplotlib.pyplot as plt
# --- TensorBoard ---
from torch.utils.tensorboard import SummaryWriter
import time
from BoatRaceDataset_base import BoatRaceDatasetBase     # ← MTL 対応版
from DualHeadRanker import DualHeadRanker
import itertools

# --- reproducibility helpers ---
import random  # reproducibility helpers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------
# Feature‑engineering registry (declarative “add / drop” infrastructure)
# ----------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Callable, Sequence, Dict
import pandas as pd  # already imported above, but kept for clarity

@dataclass
class FeatureDef:
    """Declarative feature definition."""
    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    deps: Sequence[str] = field(default_factory=tuple)  # for documentation
    dtype: str = None                            # optional cast

FEATURE_REGISTRY: Dict[str, FeatureDef] = {}

def register_feature(fd: FeatureDef):
    """Add a feature definition to the global registry."""
    FEATURE_REGISTRY[fd.name] = fd

def apply_features(
    df: pd.DataFrame,
    include: Sequence[str] = None,
    exclude: Sequence[str] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    names = include if include is not None else list(FEATURE_REGISTRY)
    if exclude:
        names = [n for n in names if n not in exclude]

    for n in names:
        fd = FEATURE_REGISTRY[n]
        df[n] = fd.fn(df)
        if fd.dtype:
            df[n] = df[n].astype(fd.dtype)
    return df

# --------------------------- default features --------------------------
def _wind_sin(df: pd.DataFrame) -> pd.Series:
    """Sine of wind direction (deg → rad)."""
    return np.sin(np.deg2rad(df["wind_dir_deg"]))

def _wind_cos(df: pd.DataFrame) -> pd.Series:
    """Cosine of wind direction (deg → rad)."""
    return np.cos(np.deg2rad(df["wind_dir_deg"]))


register_feature(FeatureDef("wind_sin", _wind_sin, deps=["wind_dir_deg"]))
register_feature(FeatureDef("wind_cos", _wind_cos, deps=["wind_dir_deg"]))


# In[ ]:


import nbformat
from nbconvert import PythonExporter

with open("main_base.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

with open("main_base.py", "w", encoding="utf-8") as f:
    f.write(source)


# In[ ]:


load_dotenv(override=True)
venue = '若 松'

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
result_df = pd.read_sql(f"""
    SELECT * FROM feat.train_features_base
    WHERE race_date <= '2024-12-31'
    AND venue = '{venue}'
""", conn)


print(f"Loaded {len(result_df)} rows from the database.")


# In[ ]:


result_df = apply_features(result_df)
# 重要列の drop バグ修正：bf_course / bf_st_time / weight は保持する
# 重要列の drop バグ修正：bf_course / bf_st_time / weight は保持する
exclude = []  # ← 重要列はドロップしない（必要があればここに追加）
for lane in range(1, 7):
      # --- 対象列を決める（ターゲット & キー列は除外） ---
      exclude.append(
            f"lane{lane}_bf_course",
      )
      exclude.append(f"lane{lane}_bf_st_time")
      exclude.append(f"lane{lane}_weight")

result_df.drop(columns=exclude, inplace=True, errors="ignore")

# numeric columns for StandardScaler
BASE_NUM_COLS = ["air_temp", "wind_speed", "wave_height",
                 "water_temp", "wind_sin", "wind_cos"]
# automatically pick up newly merged rolling features (suffix *_30d)
HIST_NUM_COLS = [c for c in result_df.columns
                 if c.endswith("_30d") and result_df[c].dtype != "object"]
NUM_COLS = BASE_NUM_COLS + HIST_NUM_COLS
print(f"[info] StandardScaler will use {len(NUM_COLS)} numeric cols "
      f"({len(BASE_NUM_COLS)} base + {len(HIST_NUM_COLS)} hist)")
scaler = StandardScaler().fit(result_df[NUM_COLS])
result_df[NUM_COLS] = scaler.transform(result_df[NUM_COLS])

bool_cols = [c for c in result_df.columns if c.endswith("_fs_flag")]
result_df[bool_cols] = result_df[bool_cols].fillna(False).astype(bool)
result_df.to_csv("artifacts/train_features_base.csv", index=False)
display(result_df.head())
print("データフレーム全体の欠損値の総数:", result_df.isnull().sum().sum())

# 各列の欠損値の割合を表示（0〜1の値）
missing_ratio = result_df.isnull().mean()

# パーセント表示にする場合（見やすさのため）
missing_ratio_percent = missing_ratio * 100

print("各列の欠損値の割合（%）:")
print(missing_ratio_percent.sort_values(ascending=False))

os.makedirs("artifacts", exist_ok=True)
scaler_filename = "artifacts/wind_scaler.pkl"
joblib.dump(scaler, scaler_filename)


# In[ ]:


def encode(col):
    uniq = sorted(result_df[col].dropna().unique())
    mapping = {v:i for i,v in enumerate(uniq)}
    result_df[col + "_id"] = result_df[col].map(mapping).fillna(-1).astype("int16")
    return mapping
venue2id = encode("venue")
# race_type2id = encode("race_type")


# In[ ]:


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
# peek_one(result_df)
# ============================================================


# ---------------- Loss / Regularization Weights -----------------
LAMBDA_ST = 0.1      # weight for ST‑MSE  (was 0.3)
L1_ALPHA  = 0.02     # weight for rank‑L1 loss
CLIP_NORM = 10.0     # gradient‑clipping threshold (was 5.0)
RANKNET_ALPHA = 0.10   # weight for pairwise RankNet loss
TEMPERATURE   = 0.80   # logits are divided by T at inference
LAMBDA_WIN = 1.0        # weight for winner‑BCE loss

TOPK_K = 3
TOPK_WEIGHTS = [3.0, 2.0, 1.0]


# In[ ]:


def pl_nll(scores: torch.Tensor, ranks: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    scores = scores.clamp(-20.0, 20.0)        # avoid Inf/NaN

    order = torch.argsort(ranks, dim=1)       # (B,6) winner→last
    nll = torch.zeros(scores.size(0), device=scores.device)
    s = scores.clone()

    for pos in range(6):
        log_denom = torch.logsumexp(s, dim=1)                 # (B,)
        idx = order[:, pos]                                   # (B,)
        chosen = s.gather(1, idx.unsqueeze(1)).squeeze(1)     # (B,)
        nll += log_denom - chosen
        s = s.scatter(1, idx.unsqueeze(1), float('-inf'))

    return nll.mean() if reduce else nll

def pl_nll_topk(scores: torch.Tensor,
                ranks: torch.Tensor,
                k: int = 3,
                weights=None,
                reduce: bool = True) -> torch.Tensor:
    """
    Top-k Plackett–Luce negative log-likelihood.
   k: number of leading positions (default 3).
    weights: length-k list/1D Tensor of non-negative weights; if None → uniform 1.0
    reduce: if False, returns per-sample loss (B,); otherwise mean.
    """
    # numeric stability
    scores = scores.clamp(-20.0, 20.0)
    B, C = scores.shape
    k = int(min(max(k, 1), C))

    if weights is None:
        w = torch.ones(k, device=scores.device, dtype=scores.dtype)
    else:
        w = torch.as_tensor(weights, device=scores.device, dtype=scores.dtype)
        if w.numel() != k:
            w = torch.ones(k, device=scores.device, dtype=scores.dtype)

    order = torch.argsort(ranks, dim=1)   # (B,6) winner→
    s = scores.clone()
    nll = torch.zeros(B, device=scores.device, dtype=scores.dtype)

    for t in range(k):
        log_denom = torch.logsumexp(s, dim=1)             # (B,)
        idx = order[:, t]                                  # (B,)
        chosen = s.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)
        nll = nll + w[t] * (log_denom - chosen)
        s = s.scatter(1, idx.unsqueeze(1), float('-inf'))  # mask the chosen lane

    nll = nll / w.sum()
    return nll.mean() if reduce else nll

# --- Pairwise RankNet loss ---
def ranknet_loss(scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """
    Pairwise RankNet loss (cross‑entropy on all lane pairs).
    ranks : (B,6) with 1=best … 6=worst.
    """
    pair_idx = list(itertools.combinations(range(6), 2))
    loss_acc = 0.0
    for i, j in pair_idx:
        S_ij = torch.sign(ranks[:, j] - ranks[:, i])  # +1 if i<j (i better)
        diff = scores[:, i] - scores[:, j]
        loss_acc += torch.nn.functional.softplus(-S_ij * diff).mean()
    return loss_acc / len(pair_idx)

# ── pl_nll が正しいか 3 秒で判定 ──
scores = torch.tensor([[6, 5, 4, 3, 2, 1]], dtype=torch.float32)  # lane0 が最強
ranks  = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)    # lane0 が 1 着
print("pl_nll should be ~0 :", pl_nll(scores, ranks).item())
print("pl_nll_topk (k=3) should be ~0 :", pl_nll_topk(scores, ranks, k=TOPK_K, weights=TOPK_WEIGHTS).item())


# In[ ]:


result_df["race_date"] = pd.to_datetime(result_df["race_date"]).dt.date
latest_date = result_df["race_date"].max()
cutoff = latest_date - dt.timedelta(days=90)

mode = "zscore"  # レース内zスコア運用
ds_train = BoatRaceDatasetBase(result_df[result_df["race_date"] <  cutoff])
ds_val   = BoatRaceDatasetBase(result_df[result_df["race_date"] >= cutoff])

loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
loader_val   = DataLoader(ds_val,   batch_size=512)

# ------------------- ⑤ 学習ループ（LR↓ + Clip） --------------
device = "cuda" if torch.cuda.is_available() else "cpu"
boat_dim = ds_train.boat_dim
print("boat_dim =", ds_train.boat_dim)
print("has_motor_rates =", getattr(ds_train, "has_motor_rates", None),
      "has_boat_rates =", getattr(ds_train, "has_boat_rates", None))
model = DualHeadRanker(boat_in=boat_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)


# In[ ]:


def evaluate_model(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=512)
    total_loss = 0
    with torch.no_grad():
        # 6 要素を受け取り、ST は無視
        for ctx, boats, lane_ids, ranks, *_ in loader:
            ctx, boats = ctx.to(device), boats.to(device)
            lane_ids, ranks = lane_ids.to(device), ranks.to(device)

            _, scores, _ = model(ctx, boats, lane_ids)   # ST & win logits are ignored
            loss = pl_nll_topk(scores, ranks, k=TOPK_K, weights=TOPK_WEIGHTS)
            total_loss += loss.item() * len(ctx)
    return total_loss / len(dataset)


# def run_experiment(data_frac, df_full, mode="zscore", epochs=5, device="cpu"):
#     df_frac = df_full.sample(frac=data_frac, random_state=42)
#     df_frac["race_date"] = pd.to_datetime(df_frac["race_date"]).dt.date
#     latest_date = df_frac["race_date"].max()
#     cutoff = latest_date - dt.timedelta(days=90)  # last 3 months used as validation set
#     ds_train = BoatRaceDatasetBase(df_frac[df_frac["race_date"] < cutoff])
#     ds_val = BoatRaceDatasetBase(df_frac[df_frac["race_date"] >= cutoff])

#     loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
#     loader_val = DataLoader(ds_val, batch_size=512)

#     boat_dim = ds_train.boat_dim
#     model = DualHeadRanker(boat_in=boat_dim).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)

#     for epoch in range(epochs):
#         model.train()
#         for ctx, boats, lane_ids, ranks, *_ in loader_train:
#             ctx, boats = ctx.to(device), boats.to(device)
#             lane_ids, ranks = lane_ids.to(device), ranks.to(device)

#             _, scores, _ = model(ctx, boats, lane_ids)        # discard ST & win head
#             loss = (pl_nll(scores, ranks, reduce=False) *
#                     torch.where(ranks[:,0]==1,
#                                 torch.tensor(1.0, device=ranks.device),
#                                 torch.tensor(1.5, device=ranks.device))).mean()
#             opt.zero_grad(); loss.backward(); opt.step()

#     train_loss = evaluate_model(model, ds_train, device)
#     val_loss = evaluate_model(model, ds_val, device)
#     return train_loss, val_loss

# # 学習曲線の描画
# def plot_learning_curve(df, device):
#     data_fracs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
#     results = []

#     for frac in data_fracs:
#         tr_loss, val_loss = run_experiment(frac, df, device=device)
#         print(f"Data frac {frac:.2f} → Train: {tr_loss:.4f} / Val: {val_loss:.4f}")
#         results.append((frac, tr_loss, val_loss))

#     fracs, tr_losses, val_losses = zip(*results)
#     plt.plot(fracs, tr_losses, label="Train Loss")
#     plt.plot(fracs, val_losses, label="Val Loss")
#     plt.xlabel("Training Data Fraction")
#     plt.ylabel("Loss")
#     plt.title("Learning Curve")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def overfit_tiny(df: pd.DataFrame, device: str = "cpu"):
#     """
#     データセットを 10 行だけに縮小し、500 step で過学習できるか検証
#     """
#     tiny_df = df.sample(10, random_state=1).reset_index(drop=True)
#     tiny_ds = BoatRaceDatasetBase(tiny_df)
#     tiny_loader = DataLoader(tiny_ds, batch_size=10, shuffle=True)

#     net = DualHeadRanker().to(device)
#     opt = torch.optim.AdamW(net.parameters(), lr=3e-3)

#     for _ in range(500):
#         ctx, boats, lane_ids, ranks, st_true, st_mask = next(iter(tiny_loader))
#         ctx, boats = ctx.to(device), boats.to(device)
#         lane_ids, ranks = lane_ids.to(device), ranks.to(device)
#         st_true, st_mask = st_true.to(device), st_mask.to(device)
#         st_pred, scores, _ = net(ctx, boats, lane_ids)
#         pl_loss = (pl_nll(scores, ranks, reduce=False) *
#                    torch.where(ranks[:,0]==1,
#                                torch.tensor(1.0, device=ranks.device),
#                                torch.tensor(1.5, device=ranks.device))).mean()
#         mse_st = ((st_pred - st_true) ** 2 * st_mask.float()).sum() / st_mask.float().sum()
#         loss = pl_loss + LAMBDA_ST * mse_st
#         opt.zero_grad(); loss.backward(); opt.step()

#     print("[tiny] final loss:", loss.item())

# RUN_DIAG = True

# if RUN_DIAG:
#     print("[diag] Running learning‑curve vs. data fraction …")
#     plot_learning_curve(result_df, device)
#     print("[diag] Running 10‑row overfit_tiny() …")
#     overfit_tiny(result_df, device)
#     print("[diag]   ► finished quick diagnostics\n")


# In[ ]:


# ---------------------------------------------------------------------
# print(f"train: {len(ds_train)}  val: {len(ds_val)}"

EPOCHS = 20
# --- TensorBoard setup ---
log_dir = os.path.join("artifacts", "tb", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
for epoch in range(EPOCHS):
    if epoch == 0:                  # 1 エポック目だけ試す例
        ctx, boats, lane_ids, ranks, st_true, st_mask = next(iter(loader_train))
        ctx, boats = ctx.to(device), boats.to(device)
        lane_ids = lane_ids.to(device)
        st_true, st_mask = st_true.to(device), st_mask.to(device)

        st_pred, scores, win_logits = model(ctx, boats, lane_ids)
        (st_pred.sum() + scores.sum() + win_logits.sum()).backward()     # ダミー backward
        grad_norm = sum(p.grad.abs().mean().item() for p in model.parameters())
        # print(f"[debug] average |grad| = {grad_norm:.3e}")
    # ---- train ----
    model.train(); tr_sum = 0
    grad_sum, grad_steps = 0.0, 0
    for ctx, boats, lane_ids, ranks, st_true, st_mask in loader_train:
        ctx, boats = ctx.to(device), boats.to(device)
        lane_ids, ranks = lane_ids.to(device), ranks.to(device)
        st_true, st_mask = st_true.to(device), st_mask.to(device)

        st_pred, scores, win_logits = model(ctx, boats, lane_ids)
        loss_each = pl_nll_topk(scores, ranks, k=TOPK_K, weights=TOPK_WEIGHTS, reduce=False)  # (B,)
        sample_weight = torch.where(ranks[:, 0] == 1,               # lane1 winner?
                                    torch.tensor(1.0, device=ranks.device),
                                    torch.tensor(1.5, device=ranks.device))
        pl_loss = (loss_each * sample_weight).mean()
        mse_st = ((st_pred - st_true) ** 2 * st_mask.float()).sum() / st_mask.float().sum()
        pred_rank = scores.argsort(dim=1, descending=True).argsort(dim=1) + 1  # 1〜6 着になるよう変換
        l1_loss = nn.L1Loss()(pred_rank.float(), ranks.float())
        rn_loss = ranknet_loss(scores, ranks)
        winner_true = (ranks == 1).float()            # one‑hot (B,6)
        bce_win = nn.BCEWithLogitsLoss()(win_logits, winner_true)
        loss = pl_loss + LAMBDA_ST * mse_st + L1_ALPHA * l1_loss + \
               RANKNET_ALPHA * rn_loss + LAMBDA_WIN * bce_win
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)  # ★勾配爆発対策★
        opt.step()

        # ---- gradient magnitude tracking ----
        g_tot, g_cnt = 0.0, 0
        for p in model.parameters():
            if p.grad is not None:
                g_tot += p.grad.detach().abs().mean().item()
                g_cnt += 1
        grad_sum += g_tot / max(g_cnt, 1)
        grad_steps += 1

        tr_sum += loss.item() * len(ctx)

    tr_nll = tr_sum / len(loader_train.dataset)

    # ---- validation ----
    model.eval(); val_sum = 0
    # --- validation: also compute st MSE/MAE
    mse_st_accum, mae_st_accum, n_st = 0.0, 0.0, 0.0
    with torch.no_grad():
        for ctx, boats, lane_ids, ranks, st_true, st_mask in loader_val:
            ctx, boats = ctx.to(device), boats.to(device)
            lane_ids, ranks = lane_ids.to(device), ranks.to(device)
            st_true, st_mask = st_true.to(device), st_mask.to(device)
            st_pred, scores, _ = model(ctx, boats, lane_ids)
            pl_loss = pl_nll_topk(scores, ranks, k=TOPK_K, weights=TOPK_WEIGHTS)
            # ST MSE
            mse_st = ((st_pred - st_true) ** 2 * st_mask.float()).sum() / st_mask.float().sum()
            # ST MAE
            abs_err = (st_pred - st_true).abs() * st_mask.float()
            mae_st = abs_err.sum() / st_mask.float().sum()
            # accumulate for epoch
            mse_st_accum += ( ((st_pred - st_true) ** 2) * st_mask.float() ).sum().item()
            mae_st_accum += abs_err.sum().item()
            n_st += st_mask.float().sum().item()
            pred_rank = scores.argsort(dim=1, descending=True).argsort(dim=1) + 1
            l1_loss = nn.L1Loss()(pred_rank.float(), ranks.float())
            rn_loss = ranknet_loss(scores, ranks)
            total = pl_loss + LAMBDA_ST * mse_st + L1_ALPHA * l1_loss + RANKNET_ALPHA * rn_loss
            val_sum += total.item() * len(ctx)

    val_nll = val_sum / len(loader_val.dataset)
    # epoch-wise st metrics
    st_mse_val = mse_st_accum / n_st if n_st > 0 else float('nan')
    st_mae_val = mae_st_accum / n_st if n_st > 0 else float('nan')

    avg_grad = grad_sum / max(grad_steps, 1)
    writer.add_scalar("diag/avg_grad", avg_grad, epoch)
    print(f"epoch {epoch:2d}  train_nll {tr_nll:.4f}  val_nll {val_nll:.4f}  |grad| {avg_grad:.2e}")
    print(f"ST MSE: {st_mse_val:.4f}  ST MAE: {st_mae_val:.4f}")

    # ---- accuracy & 三連単的中率 ----
    def top1_accuracy(scores, ranks):
        pred_top1 = scores.argmax(dim=1)
        true_top1 = (ranks == 1).nonzero(as_tuple=True)[1]
        return (pred_top1 == true_top1).float().mean().item()

    def trifecta_hit_rate(scores, ranks):
        """
        三連単的中率（予測スコア上位3頭の順番が、実際の1〜3着と完全一致する割合）
        """
        pred_top3 = torch.topk(scores, k=3, dim=1).indices
        true_top3 = torch.topk(-ranks, k=3, dim=1).indices  # 小さい順に1〜3着
        hit = [p.tolist() == t.tolist() for p, t in zip(pred_top3, true_top3)]
        return sum(hit) / len(hit)

    # accuracy 評価
    model.eval(); all_scores, all_ranks = [], []
    with torch.no_grad():
        for ctx, boats, lane_ids, ranks, _, _ in loader_val:
            ctx, boats = ctx.to(device), boats.to(device)
            lane_ids = lane_ids.to(device)
            _, scores, _ = model(ctx, boats, lane_ids)
            all_scores.append(scores.cpu())
            all_ranks.append(ranks)

    all_scores = torch.cat(all_scores, dim=0)
    all_ranks = torch.cat(all_ranks, dim=0)

    acc_top1 = top1_accuracy(all_scores, all_ranks)
    acc_tri3 = trifecta_hit_rate(all_scores, all_ranks)
    writer.add_scalar("loss/train_nll",  tr_nll,  epoch)
    writer.add_scalar("loss/val_nll",    val_nll, epoch)
    writer.add_scalar("acc/top1",        acc_top1, epoch)
    writer.add_scalar("acc/trifecta_hit", acc_tri3, epoch)
    # --- ST metrics to TensorBoard
    writer.add_scalar("st/mse", st_mse_val, epoch)
    writer.add_scalar("st/mae", st_mae_val, epoch)

    # --- スコア分散と三連単ランク分析 ---
    def score_confidence_analysis(scores: torch.Tensor) -> torch.Tensor:
        return scores.var(dim=1)

    def get_trifecta_rank(scores: torch.Tensor, true_ranks: torch.Tensor) -> list:
        from itertools import permutations
        B = scores.size(0)
        results = []

        for b in range(B):
            score = scores[b]
            true_rank = true_ranks[b]
            true_top3 = [i for i, r in enumerate(true_rank.tolist()) if r <= 3]
            true_top3_sorted = [x for _, x in sorted(zip(true_rank[true_top3], true_top3))]

            trifecta_list = list(permutations(range(6), 3))
            trifecta_scores = [(triplet, score[list(triplet)].sum().item()) for triplet in trifecta_list]
            trifecta_scores.sort(key=lambda x: x[1], reverse=True)

            for rank_idx, (triplet, _) in enumerate(trifecta_scores):
                if list(triplet) == true_top3_sorted:
                    results.append(rank_idx + 1)
                    break
            else:
                results.append(121)
        return results

    score_vars = score_confidence_analysis(all_scores)
    tri_ranks = get_trifecta_rank(all_scores, all_ranks)
    mean_var = score_vars.mean().item()
    median_var = score_vars.median().item()
    mean_tri_rank = np.mean(tri_ranks)

    # print(f"スコア分散の平均: {mean_var:.4f}")
    # print(f"スコア分散の中央値: {median_var:.4f}")
    # print(f"正解三連単の予測ランク（平均）: {mean_tri_rank:.2f}")

    writer.add_scalar("score_variance/mean", mean_var, epoch)
    writer.add_scalar("score_variance/median", median_var, epoch)
    writer.add_scalar("trifecta_rank/mean", mean_tri_rank, epoch)

    # print(f"Top-1 Acc: {acc_top1:.3f}   Trifecta Hit: {acc_tri3:.3f}")

    # ------------------------------------------------------------------
    #  Export raw softmax probabilities (6‑class) + winner label (val set)
    # ------------------------------------------------------------------
    if epoch == EPOCHS - 1:   # export once after final epoch
        print("[export] Saving raw softmax probabilities for calibration …")
        with torch.no_grad():
            ctx_all, boats_all, lane_ids_all, ranks_all = [], [], [], []
            for ctx, boats, lane_ids, ranks, *_ in loader_val:
                ctx_all.append(ctx); boats_all.append(boats)
                lane_ids_all.append(lane_ids); ranks_all.append(ranks)

            ctx_all   = torch.cat(ctx_all).to(device)
            boats_all = torch.cat(boats_all).to(device)
            lane_ids_all = torch.cat(lane_ids_all).to(device)
            ranks_all = torch.cat(ranks_all).to(device)

            _, score_all, _ = model(ctx_all, boats_all, lane_ids_all)
            prob_all = torch.softmax(score_all / TEMPERATURE, dim=1)  # (N,6)

            # winner label: 0..5
            winner_idx = (ranks_all == 1).nonzero(as_tuple=True)[1].cpu().numpy()

            df_probs = pd.DataFrame(prob_all.cpu().numpy(),
                                    columns=[f"prob_lane{i}" for i in range(1,7)])
            df_probs["winner"] = winner_idx
            probs_path = "artifacts/raw_probs_val.csv"

    # ---- 学習ログを CSV へ追記保存 ----
    import csv
    os.makedirs("artifacts", exist_ok=True)
    log_path = f"artifacts/train_{mode}.csv"
    # 1回目だけヘッダーを書き込む
    write_header = epoch == 0 and not os.path.exists(log_path)
    with open(log_path, mode="a", newline="") as f:
        writer_csv = csv.writer(f)
        if write_header:
            writer_csv.writerow(["epoch", "train_nll", "val_nll", "top1_acc", "trifecta_hit",
                                 "score_var_mean", "score_var_median", "tri_rank_mean",
                                 "st_mse", "st_mae"])
        writer_csv.writerow([epoch, tr_nll, val_nll, acc_top1, acc_tri3,
                             mean_var, median_var, mean_tri_rank,
                             st_mse_val, st_mae_val])


# --- Close TensorBoard writer after training ---
writer.close()

# modelの保存
now = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs("artifacts/models", exist_ok=True)
model_path = f"artifacts/models/model_{now}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


# In[ ]:


# ---- Monkey‑patch ROIAnalyzer so it uses BoatRaceDataset2 (MTL) ----------
from types import MethodType
from BoatRaceDataset_base import BoatRaceDatasetBase 
from torch.utils.data import DataLoader
from roi_util import ROIAnalyzer


#  # 最新のモデルを取得
# model_list = os.listdir("artifacts/models")
# model_list = [f for f in model_list if f.endswith(".pth")]
# if model_list:
#     latest_model = sorted(model_list)[-1]  # 最新のモデルを選択
#     model_path = os.path.join("artifacts", "models", latest_model)
#     print(f"Using latest model: {model_path}")
#     # モデルをロード
#     model = DualHeadRanker(boat_in=boat_dim)
#     model.load_state_dict(torch.load(model_path, map_location=device))

today = dt.date.today()
# 2025年1月1日以降のデータを取得する場合は、以下の行を変更してください。
start_date = dt.date(2025, 1, 1)
# start_date = today - dt.timedelta(days=20)

query = f"""
    SELECT * FROM feat.eval_features_base
    WHERE race_date BETWEEN '{start_date}' AND '{today}'
    AND venue = '{venue}'
"""
df_recent = pd.read_sql(query, conn)
print(df_recent)


df_recent.drop(columns=exclude, inplace=True, errors="ignore")

df_recent.to_csv("artifacts/eval_features_recent_base.csv", index=False)

if df_recent.empty:
    print("[simulate] No rows fetched for last 3 months.")

print(f"[simulate] Loaded {len(df_recent)} rows ({start_date} – {today}).")
print(f"columns: {', '.join(df_recent.columns)}")


# In[ ]:


class _RankOnly(nn.Module):
    """Adapter: forward() returns rank_pred tensor only, temperature-scaled."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, *args, **kwargs):
        _, rank_pred, _ = self.base(*args, **kwargs)
        return rank_pred / TEMPERATURE
rank_model = _RankOnly(model).to(device)

analyzer = ROIAnalyzer(model=rank_model, scaler=scaler,
                       num_cols=NUM_COLS, device=device)

# --- 予測でも「自信度」と「正解三連単の順位」を評価し、CSV に記録 ---
print("[predict] Evaluating confidence & trifecta rank on recent predictions…")

# ROIAnalyzer の前処理（スケーリング等）をそのまま使ってローダを作成
loader_eval, _df_eval_proc, _df_odds = analyzer._create_loader(df_recent)

# 既に上で用意した rank_model は「rank_pred だけ」を返すアダプタ
model.eval(); rank_model.eval()
all_scores, all_ranks, all_keys = [], [], []

print(_df_odds)

row_ptr = 0
with torch.no_grad():
    for ctx, boats, lane_ids, ranks, _, __ in loader_eval:
        ctx, boats, lane_ids = ctx.to(device), boats.to(device), lane_ids.to(device)
        scores = rank_model(ctx, boats, lane_ids)
        B = scores.size(0)

        # --- core outputs ---
        all_scores.append(scores.cpu())
        all_ranks.append(ranks)

        # --- meta values (race_key / odds) ---
        all_keys.extend(_df_eval_proc["race_key"].iloc[row_ptr : row_ptr + B].tolist())
        row_ptr += B


all_scores = torch.cat(all_scores, dim=0)   # (N,6)
all_ranks  = torch.cat(all_ranks,  dim=0)   # (N,6)


# In[ ]:


# --------------------------------------------------------------------------
#  グループ Ablation: 重要列を 5～6 個まとめてドロップして val_nll を比較
# --------------------------------------------------------------------------



def permute_importance(model, dataset, device="cpu", cols=None):
    """
    Permutation importance: 各特徴量列をランダムに permute して val_nll の悪化量を調べる
    """
    base_loss = evaluate_model(model, dataset, device)

    # ----- 列リストを決める --------------------------------------------------
    # cols=None なら「データフレームに存在する “使えそうな” 全列」を対象にする
    if cols is None:
        # 予測ターゲットやキー列は除外
        skip = {"race_key", "race_date"}
        # rank 列（教師信号）や欠損だらけの列も除外
        skip |= {c for c in dataset.f.columns if c.endswith("_rank")}
        cols = [c for c in dataset.f.columns if c not in skip]

    importances: dict[str, float] = {}
    df_full = dataset.f

    for col in cols:
        # --- その列だけランダムに permute ---
        shuffled = df_full.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        tmp_ds = BoatRaceDatasetBase(shuffled)
        loss = evaluate_model(model, tmp_ds, device)
        importances[col] = loss - base_loss   # 悪化分 (大 → 重要)
    return importances

def run_ablation_groups(
    df_full: pd.DataFrame,
    group_size: int = 6,
    epochs: int = 5,
    seed: int = 42,
    device: str = "cpu",
):
    """
    全特徴量をランダムに group_size 個ずつ束ね、
    そのグループを丸ごと削除して再学習 → val_nll を返す。

    戻り値: list[tuple[list[str], float]]
        (ドロップした列リスト, val_nll) を val_nll 昇順で並べたもの
    """
    random.seed(seed)

    essential_cols = set(NUM_COLS)          # ctx 用の連続値
    for l in range(1, 7):
        essential_cols.update({
            f"lane{l}_exh_time",
            f"lane{l}_st",
            f"lane{l}_weight",
            f"lane{l}_bf_course",
            f"lane{l}_fs_flag",
            f"lane{l}_racer_id",
            f"lane{l}_racer_name",
            f"lane{l}_racer_age",
            f"lane{l}_racer_weight",
        })
    # --- 対象列を決める（ターゲット & キー列は除外） ---
    skip = {"race_key", "race_date"}
    skip |= {c for c in df_full.columns if c.endswith("_rank")}
    skip |= essential_cols  
    skip |= {c for c in df_full.columns if c.endswith("_rank")}
    cols = [c for c in df_full.columns if c not in skip]
    random.shuffle(cols)

    groups = [cols[i : i + group_size] for i in range(0, len(cols), group_size)]
    results = []

    latest_date = pd.to_datetime(df_full["race_date"]).dt.date.max()
    cutoff = latest_date - dt.timedelta(days=90)

    for g in groups:
        df_drop = df_full.drop(columns=g)

        ds_tr = BoatRaceDatasetBase(df_drop[df_drop["race_date"] < cutoff])
        ds_va = BoatRaceDatasetBase(df_drop[df_drop["race_date"] >= cutoff])

        ld_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
        ld_va = DataLoader(ds_va, batch_size=512)

        model = DualHeadRanker().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)

        for _ in range(epochs):
            model.train()
            for ctx, boats, lane_ids, ranks, st_true, st_mask in ld_tr:
                ctx, boats = ctx.to(device), boats.to(device)
                lane_ids, ranks = lane_ids.to(device), ranks.to(device)
                
                st_true, st_mask = st_true.to(device), st_mask.to(device)
                st_pred, scores = model(ctx, boats, lane_ids)
                pl_loss = pl_nll_topk(scores, ranks, k=TOPK_K, weights=TOPK_WEIGHTS)
                mse_st = ((st_pred - st_true) ** 2 * st_mask.float()).sum() / st_mask.float().sum()
                loss = pl_loss + LAMBDA_ST * mse_st
                opt.zero_grad(); loss.backward(); opt.step()

        val_loss = evaluate_model(model, ds_va, device)
        results.append((g, val_loss))

    return sorted(results, key=lambda x: x[1])  # 小さい順に重要

print("▼ Permutation importance (ALL features)")
all_imp = permute_importance(model, ds_val, device)
imp_path = "artifacts/perm_importance_all_base.csv"
pd.Series(all_imp).sort_values(ascending=False).to_csv(imp_path)
print(f"[saved] {imp_path}")


# In[ ]:


# all_ranksとall_scoresを結合したdfに変換
df_scores = pd.DataFrame(all_scores.numpy(), columns=[f"lane{i+1}_score" for i in range(6)])
df_ranks = pd.DataFrame(all_ranks.numpy(), columns=[f"lane{i+1}_rank" for i in range(6)])
df_score_ranks = pd.concat([df_scores, df_ranks], axis=1)   
df_score_ranks["race_key"] = all_keys

# df_mergedから重複行を削除
df_score_ranks = df_score_ranks.drop_duplicates()

# merge odds from df_trifecta_met_hit by race_key
df_score_ranks = df_score_ranks.merge(_df_odds[["race_key","trifecta_odds"]], on="race_key", how="left")

# --- lane 列をまとめて list 化 ---
score_cols = [f"lane{i}_score" for i in range(1, 7)]
rank_cols  = [f"lane{i}_rank"  for i in range(1, 7)]

df_score_ranks["scores"] = df_score_ranks[score_cols].apply(
    lambda r: [float(x) for x in r.values.tolist()], axis=1
)
df_score_ranks["ranks"] = df_score_ranks[rank_cols].apply(
    lambda r: [int(x) for x in r.values.tolist()], axis=1
)

from itertools import permutations

def pl_true_order_prob(scores, ranks):
    """
    Plackett–Luce で '真の完全着順(1→6位)' の確率を計算。
    scores: 長さ6のスコア配列, ranks: 長さ6の真の順位 (1=最上位)
    """
    w = np.exp(np.array(scores, dtype=float))
    # 真の順序（1→2→…→6）に並んだインデックス
    order = [i for i, _ in sorted(enumerate(ranks), key=lambda t: t[1])]
    denom = float(w.sum())
    p = 1.0
    for idx in order:
        if denom <= 0:
            return 0.0
        p *= float(w[idx] / denom)
        denom -= float(w[idx])
    return float(p)

# 6! (=720) 通りの全順位
ALL_PERMS = list(permutations(range(6), 6))

def true_order_rank(scores, ranks):
    """
    全 6! 通りの PL 確率で並べたとき、真の完全順位が何番目か（1始まり）。
    """
    w = np.exp(np.array(scores, dtype=float))
    denom0 = float(w.sum())
    true_perm = tuple(i for i, _ in sorted(enumerate(ranks), key=lambda t: t[1]))

    def prob_of_perm(perm):
        denom = denom0
        p = 1.0
        for idx in perm:
            if denom <= 0:
                return 0.0
            p *= float(w[idx] / denom)
            denom -= float(w[idx])
        return p

    probs = [(perm, prob_of_perm(perm)) for perm in ALL_PERMS]
    probs.sort(key=lambda x: x[1], reverse=True)

    for k, (perm, _) in enumerate(probs, start=1):
        if perm == true_perm:
            return k
    return len(probs) + 1  # 通常は到達しない

# 列の追加
df_score_ranks["true_order_prob"] = df_score_ranks.apply(
    lambda row: pl_true_order_prob(row["scores"], row["ranks"]), axis=1
)
df_score_ranks["true_order_rank"] = df_score_ranks.apply(
    lambda row: true_order_rank(row["scores"], row["ranks"]), axis=1
)

# 保存
df_score_ranks.to_csv("artifacts/merged_scores_ranks_base.csv", index=False)

# df_score_ranksを行でループ
total_benefit = 0.0
total_submit = 0.0

for n in range(1, 6):
    # ★ 各 n でリセット
    total_submit = 0.0
    total_benefit = 0.0

    for _, row in df_score_ranks.iterrows():
        total_submit += 100 * n
        odds = row.get("trifecta_odds", None)
        true_rank = row.get("true_order_rank", None)

        if true_rank is not None and true_rank <= n:
            # ★ 欠損オッズは0扱い
            if pd.isna(odds):
                odds = 0.0
            total_benefit += float(odds) * 100

    roi = ((total_benefit / total_submit) * 100) if total_submit > 0 else float("nan")
    print(f"n = {n}")
    print(f"total_submit : {total_submit:.2f} JPY")
    print(f"total_benefit: {total_benefit:.2f} JPY")
    print(f"roi : {roi:.2f}%")



# In[ ]:


def top1_accuracy(scores: torch.Tensor, ranks: torch.Tensor) -> float:
    """Top‑1 accuracy: predicted winner vs true winner (ranks: 1 is best)."""
    pred_top1 = scores.argmax(dim=1)        # (B,)
    true_top1 = ranks.argmin(dim=1)         # (B,)
    return (pred_top1 == true_top1).float().mean().item()

def trifecta_hit_rate(scores: torch.Tensor, ranks: torch.Tensor) -> float:
    """
    予測スコア上位3艇の順番が、実際の1〜3着と完全一致する割合。
    """
    pred_top3 = torch.topk(scores, k=3, dim=1).indices
    true_top3 = torch.topk(-ranks, k=3, dim=1).indices  # 小さい順に 1→3 着
    hit = [p.tolist() == t.tolist() for p, t in zip(pred_top3, true_top3)]
    return float(sum(hit) / len(hit)) if len(hit) else float("nan")

def constant_123_trifecta_hit(ranks: torch.Tensor) -> float:
    """
    Hit‑rate when always predicting trifecta 1‑2‑3 in order.
    """
    true_top3 = torch.topk(-ranks, k=3, dim=1).indices   # (B,3)
    baseline  = torch.tensor([0, 1, 2], dtype=torch.long, device=ranks.device)
    return (true_top3 == baseline).all(dim=1).float().mean().item()

def baseline123_position_accuracy(ranks: torch.Tensor, pos: int) -> float:
    """
    Baseline per‑position accuracy when assuming boat pos finishes pos‑th.
    pos ∈ {1,2,3}
    """
    true_idx = (ranks == pos).float().argmax(dim=1)          # (B,)
    baseline_idx = torch.tensor(pos - 1, dtype=torch.long, device=ranks.device)
    return (true_idx == baseline_idx).float().mean().item()

def baseline123_top3_unordered_hit(ranks: torch.Tensor) -> float:
    """
    Order‑agnostic hit‑rate when always predicting the set {1,2,3}.
    """
    true_top3 = torch.topk(-ranks, k=3, dim=1).indices
    hit = [set(t.tolist()) == {0,1,2} for t in true_top3]
    return float(sum(hit) / len(hit)) if len(hit) else float("nan")

def get_trifecta_rank_unordered(scores: torch.Tensor, true_ranks: torch.Tensor) -> list[int]:
    """真の三連複（順序なし）集合が、全20集合の中で何番目か（1始まり）"""
    from itertools import combinations
    combos = list(combinations(range(6), 3))
    res: list[int] = []
    for sc, tr in zip(scores, true_ranks):
        true_set = {i for i, r in enumerate(tr.tolist()) if r <= 3}
        combo_scores = [(c, sc[list(c)].sum().item()) for c in combos]
        combo_scores.sort(key=lambda x: x[1], reverse=True)
        for idx, (c, _) in enumerate(combo_scores, start=1):
            if set(c) == true_set:
                res.append(idx)
                break
        else:
            res.append(len(combos) + 1)
    return res

def get_trifecta_rank_ordered(scores: torch.Tensor, true_ranks: torch.Tensor) -> list[int]:
    """真の三連単（順序あり）が、PL確率で並べた全120順列の中で何番目か（1始まり）。"""
    import itertools
    perms = list(itertools.permutations(range(6), 3))
    res: list[int] = []
    for sc, tr in zip(scores, true_ranks):
        # 数値安定化: 行内の最大を引いてから exp
        es = torch.exp(sc - sc.max())
        ordered_true = sorted(range(6), key=lambda i: tr[i].item())[:3]

        denom0 = es.sum().item()
        perm_probs = []
        for p0, p1, p2 in perms:
            d1 = denom0
            d2 = d1 - es[p0].item()
            d3 = d2 - es[p1].item()
            if d2 <= 0 or d3 <= 0:
                prob = 0.0
            else:
                prob = (es[p0] / d1) * (es[p1] / d2) * (es[p2] / d3)
            perm_probs.append(((p0, p1, p2), float(prob)))

        perm_probs.sort(key=lambda x: x[1], reverse=True)
        for idx, (p, _) in enumerate(perm_probs, start=1):
            if list(p) == ordered_true:
                res.append(idx)
                break
        else:
            res.append(len(perms) + 1)
    return res

def top3_unordered_hit_rate(scores: torch.Tensor, ranks: torch.Tensor) -> float:
    pred_top3 = torch.topk(scores, k=3, dim=1).indices
    true_top3 = torch.topk(-ranks, k=3, dim=1).indices
    hit = [(set(p.tolist()) == set(t.tolist())) for p, t in zip(pred_top3, true_top3)]
    return float(sum(hit) / len(hit)) if len(hit) else float("nan")

def mean_reciprocal_rank(scores: torch.Tensor, ranks: torch.Tensor) -> float:
    order = scores.argsort(dim=1, descending=True)          # (B,6)
    true_winner_idx = ranks.argmin(dim=1)                   # (B,)
    # position of true winner in each row (1-based)
    pos = (order == true_winner_idx[:, None]).float().argmax(dim=1) + 1
    return (1.0 / pos.float()).mean().item()

def spearman_corr(scores: torch.Tensor, ranks: torch.Tensor) -> float:
    """
    Average per‑race Spearman rank correlation between predicted and true rank orders.
    """
    pred_rank = scores.argsort(dim=1, descending=True).argsort(dim=1).float() + 1  # 1..6
    true_rank = ranks.float()
    d = pred_rank - true_rank
    rho = 1 - 6 * (d ** 2).sum(dim=1) / (6 * (6**2 - 1))
    return rho.mean().item()

score_vars = all_scores.var(dim=1)
tri_ranks  = get_trifecta_rank_unordered(all_scores, all_ranks)
tri_ranks_order = get_trifecta_rank_ordered(all_scores, all_ranks)
mean_tri_order  = float(np.mean(tri_ranks_order)) if len(tri_ranks_order) else float("nan")

# ---- Popularity vs Model comparison (ordered trifecta) ----
# We expect df_recent to carry 'trifecta_popularity_rank' per race_key (ordered).
pop_col = None
for c in ["trifecta_popularity_rank"]:
    if c in df_recent.columns:
        pop_col = c; break

if pop_col is not None:
    _model_rank_df = pd.DataFrame({
        "race_key": np.array(all_keys),
        "model_trifecta_rank_ordered": tri_ranks_order
    })
    _pop_df = df_recent[["race_key", pop_col]].drop_duplicates("race_key").rename(columns={pop_col: "pop_trifecta_rank_ordered"})
    _cmp_df = _model_rank_df.merge(_pop_df, on="race_key", how="inner")
    _cmp_df["pop_trifecta_rank_ordered"] = pd.to_numeric(_cmp_df["pop_trifecta_rank_ordered"], errors="coerce")
    _cmp_df = _cmp_df.dropna(subset=["pop_trifecta_rank_ordered"])  # keep valid rows only

    # Gains: positive if model ranks the true trifecta higher (smaller rank) than popularity
    _cmp_df["gain"] = _cmp_df["pop_trifecta_rank_ordered"] - _cmp_df["model_trifecta_rank_ordered"]

    pop_tri_rank_mean = float(_cmp_df["pop_trifecta_rank_ordered"].mean()) if len(_cmp_df) else float("nan")
    model_vs_pop_avg_gain = float(_cmp_df["gain"].mean()) if len(_cmp_df) else float("nan")
    model_vs_pop_median_gain = float(_cmp_df["gain"].median()) if len(_cmp_df) else float("nan")
    model_beats_pop_rate = float((_cmp_df["gain"] > 0).mean()) if len(_cmp_df) else float("nan")

    def _cov(N: int):
        if len(_cmp_df) == 0:
            return float("nan"), float("nan"), float("nan")
        pop_hit = float((_cmp_df["pop_trifecta_rank_ordered"] <= N).mean())
        model_hit = float((_cmp_df["model_trifecta_rank_ordered"] <= N).mean())
        return pop_hit, model_hit, model_hit - pop_hit

    pop_top1, model_top1_ord, diff_top1 = _cov(1)
    pop_top3, model_top3_ord, diff_top3 = _cov(3)
else:
    pop_tri_rank_mean = model_vs_pop_avg_gain = model_vs_pop_median_gain = model_beats_pop_rate = float("nan")
    pop_top1 = model_top1_ord = diff_top1 = pop_top3 = model_top3_ord = diff_top3 = float("nan")
    print("[pop] Column 'trifecta_popularity_rank' not found; skipping pop vs model comparison.")

acc_top1   = top1_accuracy(all_scores, all_ranks)
acc_tri3   = trifecta_hit_rate(all_scores, all_ranks)
mean_var   = score_vars.mean().item()
median_var = score_vars.median().item()
mean_tri   = float(np.mean(tri_ranks)) if len(tri_ranks) else float("nan")

# ---- compute new metrics ----
hit_top3_unordered = top3_unordered_hit_rate(all_scores, all_ranks)
mrr_winner        = mean_reciprocal_rank(all_scores, all_ranks)
rho_spearman      = spearman_corr(all_scores, all_ranks)

# ---- baseline metrics (constant 1‑2‑3) ----
tri123_hit      = constant_123_trifecta_hit(all_ranks)
base_pos1       = baseline123_position_accuracy(all_ranks, 1)
base_pos2       = baseline123_position_accuracy(all_ranks, 2)
base_pos3       = baseline123_position_accuracy(all_ranks, 3)
base_top1       = base_pos1                                   # same as pos1
base_top3_unord = baseline123_top3_unordered_hit(all_ranks)

# --- per-position accuracy (model) ---
def position_accuracy(ranks: torch.Tensor, scores: torch.Tensor, pos: int) -> float:
    """
    Accuracy for predicting which boat finishes pos‑th.
    """
    # Model's prediction: which boat is pos-th in predicted ranking
    pred_rank = scores.argsort(dim=1, descending=True).argsort(dim=1) + 1
    pred_idx = (pred_rank == pos).float().argmax(dim=1)
    true_idx = (ranks == pos).float().argmax(dim=1)
    return (pred_idx == true_idx).float().mean().item()

acc_pos1 = position_accuracy(all_ranks, all_scores, 1)
acc_pos2 = position_accuracy(all_ranks, all_scores, 2)
acc_pos3 = position_accuracy(all_ranks, all_scores, 3)

print(f"[predict] N={len(all_scores)}")
print(f"  • Top‑1 Acc              : {acc_top1:.3f}   (baseline {base_top1:.3f})")
print(f"  • Pos1/2/3 Acc           : {acc_pos1:.3f}/{acc_pos2:.3f}/{acc_pos3:.3f} "
      f"(baseline {base_pos1:.3f}/{base_pos2:.3f}/{base_pos3:.3f})")
print(f"  • Top‑3 unordered Hit    : {hit_top3_unordered:.3f}   (baseline {base_top3_unord:.3f})")
print(f"  • Trifecta Hit           : {acc_tri3:.3f}   (baseline {tri123_hit:.3f})")
print(f"  • Winner MRR             : {mrr_winner:.3f}")
print(f"  • Spearman ρ             : {rho_spearman:.3f}")
print(f"  • Score variance (mean/median): {mean_var:.4f} / {median_var:.4f}")
print(f"  • Avg rank of true trifecta (unordered) : {mean_tri:.2f}")
print(f"  • Avg rank of true trifecta (strict)    : {mean_tri_order:.2f}")
print(f'gain: pop_rank − model_rank')
print(f"  • Pop vs Model (ordered true trifecta rank): mean pop {pop_tri_rank_mean:.2f}, avg gain {model_vs_pop_avg_gain:.2f}, median gain {model_vs_pop_median_gain:.2f}, beat-rate {model_beats_pop_rate:.3f}")
print(f"  • TopN cover (ordered): N=1 model {model_top1_ord:.3f} vs pop {pop_top1:.3f} (Δ{diff_top1:.3f}); N=3 model {model_top3_ord:.3f} vs pop {pop_top3:.3f} (Δ{diff_top3:.3f})")

# ---- CSV に追記保存 ----
import csv, os
os.makedirs("artifacts", exist_ok=True)
metrics_path = "artifacts/predict_metrics_recent.csv"
write_header = not os.path.exists(metrics_path)
with open(metrics_path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["date", "n_races",
                    "top1_acc", "pos1_acc", "pos2_acc", "pos3_acc",
                    "top3unordered_hit", "trifecta_hit",
                    "baseline123_hit", "baseline123_top1",
                    "baseline123_pos1", "baseline123_pos2", "baseline123_pos3",
                    "baseline123_top3unordered",
                    "winner_mrr", "spearman_rho",
                    "var_mean", "var_median", "tri_rank_mean",
                    "tri_rank_order_mean",
                    "pop_tri_rank_mean", "model_vs_pop_avg_gain", "model_vs_pop_median_gain", "model_beats_pop_rate",
                    "pop_top1_hit", "model_top1_ord_hit", "delta_top1",
                    "pop_top3_hit", "model_top3_ord_hit", "delta_top3"])
    w.writerow([str(today), len(all_scores),
                acc_top1, acc_pos1, acc_pos2, acc_pos3,
                hit_top3_unordered, acc_tri3,
                tri123_hit, base_top1,
                base_pos1, base_pos2, base_pos3,
                base_top3_unord,
                mrr_winner, rho_spearman,
                mean_var, median_var, mean_tri, mean_tri_order,
                pop_tri_rank_mean, model_vs_pop_avg_gain, model_vs_pop_median_gain, model_beats_pop_rate,
                pop_top1, model_top1_ord, diff_top1,
                pop_top3, model_top3_ord, diff_top3])
print(f"[saved] {metrics_path}")


# In[ ]:


# === 条件別ヒット率/ROI 分析（修正版） =========================
# 目的: 「どんな条件のときに当たりやすいか？」を、ヒット率とROIで可視化
# 入力: df_score_ranks（race_key, trifecta_odds, true_order_rank を含む）
#       df_recent（環境・会場などの特徴）
# 依存: analyzer, model, rank_model, device がスコープに存在する想定
# 出力: artifacts/cond_base_table.csv, artifacts/cond_hit_roi.csv
# 重要: 意思決定に使う条件からは “事後情報” の疑いがあるもの（例: trifecta_odds_bin）を除外
# -------------------------------------------------------------
import os
import math
from itertools import permutations

import numpy as np
import pandas as pd
import torch

print("[cond] 条件別の当たりやすさ分析を開始…")

# 0) ベース表の構築（分析に使う列をまとめる）
_base = df_score_ranks.copy()
_base = _base[_base["race_key"].notna()].copy()
_base["true_order_rank"] = pd.to_numeric(_base["true_order_rank"], errors="coerce")
_base["trifecta_odds"] = pd.to_numeric(_base["trifecta_odds"], errors="coerce")

# df_recent から環境や会場などの列をマージ（存在する列だけ）
_cand_cols = [
    "race_key", "venue", "air_temp", "water_temp",
    "wind_speed", "wave_height", "wind_dir_deg", "wind_sin", "wind_cos",
]
_exist_cols = [c for c in _cand_cols if c in df_recent.columns]
if _exist_cols:
    _env = df_recent[_exist_cols].drop_duplicates("race_key")
    _base = _base.merge(_env, on="race_key", how="left")

# 1) 予測スコア由来の「自信度」特徴を付与（PLのtop1確率・top2とのギャップ等）
try:
    _scores_mat = all_scores.detach().cpu().numpy()  # (N,6)
    _rk_seq = _df_eval_proc["race_key"].to_numpy()
except Exception as e:
    print("[cond] all_scores が見つからない/使えないため再計算します:", e)
    loader_eval, _df_eval_proc, _ = analyzer._create_loader(df_recent)
    model.eval(); rank_model.eval()
    _sc_list = []
    with torch.no_grad():
        for ctx, boats, lane_ids, _ranks in loader_eval:
            ctx, boats, lane_ids = ctx.to(device), boats.to(device), lane_ids.to(device)
            _sc = rank_model(ctx, boats, lane_ids)
            _sc_list.append(_sc.cpu())
    all_scores = torch.cat(_sc_list, dim=0)
    _scores_mat = all_scores.detach().cpu().numpy()
    _rk_seq = _df_eval_proc["race_key"].to_numpy()

_scores_df = pd.DataFrame(_scores_mat, columns=[f"s{i}" for i in range(6)])
_scores_df["race_key"] = _rk_seq
_perms = list(permutations(range(6), 3))

def _pl_feats_from_scores(row):
    """数値安定化したsoftmax + Plackett–Luce近似で top1/top2 確率とギャップ等を算出"""
    s = np.array([row[f"s{i}"] for i in range(6)], dtype=float)
    # 数値安定化：log-sum-exp（最大値でシフト）
    s_ = s - np.max(s)
    es = np.exp(s_)
    denom0 = es.sum()

    # lane softmax のエントロピー（低いほど確信強）
    p = es / max(denom0, 1e-12)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    var = float(np.var(s))

    # 全120通りのPL確率から top1 / top2 とギャップ
    best1p, best2p = -1.0, -1.0
    best1 = None
    for a, b, c in _perms:  # 120通り
        d2 = denom0 - es[a]
        d3 = d2 - es[b]
        if d2 <= 0 or d3 <= 0:
            continue
        prob = (es[a]/denom0) * (es[b]/d2) * (es[c]/d3)  # Plackett–Luce
        if prob > best1p:
            best2p = best1p
            best1p = float(prob)
            best1 = (a, b, c)
        elif prob > best2p:
            best2p = float(prob)

    gap = best1p - best2p if best2p >= 0 else np.nan
    top1_str = f"{best1[0]+1}-{best1[1]+1}-{best1[2]+1}" if best1 is not None else np.nan
    return pd.Series({
        "pl_top1_prob": best1p,
        "pl_top2_prob": best2p,
        "pl_gap": gap,
        "pl_top1": top1_str,
        "score_entropy": entropy,
        "score_var": var,
    })

_pl_feats = _scores_df.apply(_pl_feats_from_scores, axis=1)
_scores_df = pd.concat([_scores_df[["race_key"]], _pl_feats], axis=1)
_base = _base.merge(_scores_df, on="race_key", how="left")

# 2) 条件のビニング
def _safe_qcut(series, q):
    try:
        return pd.qcut(series, q=q, duplicates="drop")
    except Exception:
        return pd.Series([np.nan] * len(series), index=series.index)

# ※ 意思決定に使う条件からは “事後情報” の trifecta_odds_bin を除外
if "pl_top1_prob" in _base.columns:
    _base["pl_prob_bin"] = _safe_qcut(_base["pl_top1_prob"], 4)
if "pl_gap" in _base.columns:
    _base["gap_bin"] = _safe_qcut(_base["pl_gap"], 4)
if "wind_speed" in _base.columns:
    _base["wind_bin"] = pd.cut(_base["wind_speed"], bins=[-np.inf, 2, 4, 6, 8, np.inf], right=False)
if "wave_height" in _base.columns:
    _base["wave_bin"] = pd.cut(_base["wave_height"], bins=[-np.inf, 0.5, 1.0, 2.0, np.inf], right=False)
if "wind_sin" in _base.columns:
    _base["tailwind"] = _base["wind_sin"] < 0  # True=追い風（sin<0）

# 3) 条件ごとのヒット率/ROI を「レース単位」で集計
def _summarize_by(col, top_n, min_races=30):
    """各レース=1票 として評価。コスト= top_n/レース、払戻= 的中レースの trifecta_odds 合計。"""
    if col not in _base.columns:
        return pd.DataFrame()
    df = _base.dropna(subset=[col, "true_order_rank", "trifecta_odds", "race_key"]).copy()
    if df.empty:
        return pd.DataFrame()

    # グループ毎に race_key でユニーク化（複製があっても1レース1回の評価にする）
    def _agg_group(g):
        rep = g.drop_duplicates("race_key")
        n_races = rep["race_key"].nunique()
        # 的中判定（「上位top_nに真の順序が入っていたか」）
        hits_mask = rep["true_order_rank"] <= top_n
        hit_rate = float(hits_mask.mean())
        # 払戻（的中レースだけカウント）
        total_return = float(rep.loc[hits_mask, "trifecta_odds"].sum())
        cost = n_races * top_n
        roi = (total_return - cost) / cost if cost > 0 else np.nan
        avg_odds_on_hits = float(rep.loc[hits_mask, "trifecta_odds"].mean()) if hits_mask.any() else np.nan
        return pd.Series({
            "n_races": n_races,
            "hit_rate": hit_rate,
            "roi": roi,
            "avg_odds_on_hits": avg_odds_on_hits,
        })

    out = df.groupby(col, dropna=False).apply(_agg_group).reset_index()
    out = out.rename(columns={col: "bin"})
    if out.empty:
        return out

    # 表示互換のため n も付ける（= n_races）
    out["n"] = out["n_races"]
    out["top_n"] = top_n
    out["condition"] = col
    out = out[out["n_races"] >= min_races].sort_values(["roi", "hit_rate"], ascending=False)
    return out

# 解析対象列（意思決定用の条件のみ）
_cols_to_try = ["pl_prob_bin", "gap_bin", "wind_bin", "wave_bin", "tailwind"]
if "venue" in _base.columns:
    _cols_to_try.append("venue")

_tables = []
for _n in [1, 2, 3, 4, 5]:
    for _c in _cols_to_try:
        if _c in _base.columns:
            _t = _summarize_by(_c, _n, min_races=30 if _c != "venue" else 50)
            if not _t.empty:
                _tables.append(_t)

_cond_result = pd.concat(_tables, ignore_index=True) if _tables else pd.DataFrame()

# 4) 保存
os.makedirs("artifacts", exist_ok=True)
_base.to_csv("artifacts/cond_base_table.csv", index=False)
_cond_result.to_csv("artifacts/cond_hit_roi.csv", index=False)

# 5) コンソールにハイライト表示
for _n in [1, 2, 3, 4, 5]:
    if not _cond_result.empty:
        with pd.option_context("display.max_rows", 20, "display.max_colwidth", 60):
            print(f"[cond] 上位の条件例 (top_n={_n}, ROI順 上位10)")
            cols_to_show = ["condition", "n", "hit_rate", "roi", "avg_odds_on_hits", "top_n"]
            # 必要に応じて各条件固有のビン列名を追記
            extra_cols = []
            # 条件ごとのビン列（表示があれば自動で含める）
            for c in ["pl_prob_bin", "gap_bin", "wind_bin", "wave_bin", "tailwind", "venue"]:
                if c in _cond_result["condition"].unique():
                    extra_cols.append(c)
            df_view = _cond_result.query(f"top_n == {_n}").sort_values("roi", ascending=False)
            # 少なくとも代表的なカラムが出るように調整
            print(df_view[["condition", "bin", "n_races", "hit_rate", "roi", "avg_odds_on_hits", "top_n"]].head(10))
    else:
        print("[cond] 条件別集計を作成できませんでした（対象列やデータ不足）。")
# ============================================================

# 4.5) ROI が最大の条件に一致するレースを CSV に出力
#  - 全体で ROI 最大の条件に一致するレース一覧
#  - top_n ごとに ROI 最大の条件に一致するレース一覧
if not _cond_result.empty:
    # 共通: 条件に一致するレースを抽出してメタ情報を付与
    def _filter_matches(row):
        col = row["condition"]
        binv = row["bin"]
        topn = int(row["top_n"])
        sel = _base.dropna(subset=["race_key"]).copy()
        # bin の型差（Categorical/Interval 等）に備えて比較をフォールバック
        try:
            mask = sel[col] == binv
        except Exception:
            mask = sel[col].astype(str) == str(binv)
        sel = sel.loc[mask].drop_duplicates("race_key").copy()

        # 出力に含める代表列（存在するものだけ採用）
        cols_pref = [
            "race_key", "trifecta_odds", "true_order_rank",
            "pl_top1_prob", "pl_top2_prob", "pl_gap", "pl_top1",
            "score_entropy", "score_var",
            "venue", "air_temp", "water_temp",
            "wind_speed", "wave_height", "wind_dir_deg", "wind_sin", "wind_cos",
        ]
        cols_exist = [c for c in cols_pref if c in sel.columns]
        if cols_exist:
            sel = sel[cols_exist]

        # メタ情報
        sel["condition"] = col
        sel["bin"] = binv
        sel["top_n"] = topn
        sel["roi_bin"] = float(row["roi"])
        sel["n_races_bin"] = int(row["n_races"])
        sel["hit_rate_bin"] = float(row["hit_rate"])
        sel["avg_odds_on_hits_bin"] = (
            float(row["avg_odds_on_hits"]) if pd.notna(row["avg_odds_on_hits"]) else np.nan
        )
        return sel

    os.makedirs("artifacts", exist_ok=True)

    # (A) 全体で ROI 最大の条件
    _best_overall = _cond_result.sort_values("roi", ascending=False).head(1)
    _best_overall_matches = _filter_matches(_best_overall.iloc[0])
    _best_overall_path = "artifacts/cond_best_roi_matches.csv"
    _best_overall_matches.to_csv(_best_overall_path, index=False)
    print(f"[cond] Best-ROI matches (overall) saved to {_best_overall_path}  —  "
          f"condition={_best_overall.iloc[0]['condition']}, bin={_best_overall.iloc[0]['bin']}, "
          f"ROI={_best_overall.iloc[0]['roi']:.3f}, top_n={int(_best_overall.iloc[0]['top_n'])}")

    # (B) top_n ごとの ROI 最大条件
    _best_by_topn = (
        _cond_result.sort_values("roi", ascending=False)
                    .groupby("top_n", as_index=False)
                    .head(1)
    )
    _dfs = []
    for _, r in _best_by_topn.iterrows():
        _dfs.append(_filter_matches(r))
    if _dfs:
        _path2 = "artifacts/cond_best_roi_matches_by_topn.csv"
        pd.concat(_dfs, ignore_index=True).to_csv(_path2, index=False)
        print(f"[cond] Best-ROI matches (per top_n) saved to {_path2}")
else:
    print("[cond] ROI 集計が空のため、best ROI CSV の出力はスキップしました。")


# In[ ]:


# prediction
from roi_util import ROIPredictor
import pandas as pd
import datetime as dt

today = dt.date.today()
# 2025年1月1日以降のデータを取得する場合は、以下の行を変更してください。
start_date = dt.date(2025, 8, 9)

query = f"""
    SELECT * FROM pred.features_with_record
    WHERE race_date BETWEEN '{start_date}' AND '{today}'
"""

conn = psycopg2.connect(**DB_CONF)
df_recent = pd.read_sql(query, conn)
print(df_recent)
df_recent.to_csv("artifacts/pred_features_recent.csv", index=False)


df_recent.drop(columns=exclude, inplace=True, errors="ignore")

if df_recent.empty:
    print("[predict] No rows fetched for the specified period.")

print(f"[predict] Loaded {len(df_recent)} rows ({start_date} – {today}).")
print(f"columns: {', '.join(df_recent.columns)}")

# ------------------------------
# ROIPredictor でスコア＆確率を一括生成
# ------------------------------
predictor = ROIPredictor(model=rank_model, scaler=scaler,
                         num_cols=NUM_COLS, device=device, batch_size=512)

# (1) スコア（logits）: lane1_score..lane6_score (+ メタ列) を保存
pred_scores_df = predictor.predict_scores(df_recent,
                                          include_meta=True,
                                          save_to="artifacts/pred_scores.csv")
display(pred_scores_df.head())


# (2) 勝率＆フェアオッズを保存
pred_probs_df = predictor.predict_win_probs(scores_df=pred_scores_df,
                                            include_meta=True,
                                            save_to="artifacts/pred_win_probs.csv")
display(pred_probs_df.head())

# (3) 馬単/三連単の TOP‑K（PL 方式）を保存
exa_df, tri_df = predictor.predict_exotics_topk(scores_df=pred_scores_df,
                                                K=10,
                                                tau=5.0,
                                                include_meta=True,
                                                save_exacta="artifacts/pred_exacta_topk.csv",
                                                save_trifecta="artifacts/pred_trifecta_topk.csv")

# (4) レース単位の PL top‑1 三連単確率を算出して書き出し／付与
import numpy as np
from itertools import permutations

def _pl_top1_from_scores_row(row: pd.Series) -> pd.Series:
    """
    lane1_score..lane6_score から
      - pl_top1_prob: 120通りのPL確率のうち最大（top‑1三連単の確率）
      - pl_top2_prob: 2番目のPL確率
      - pl_gap      : 両者の差（自信度の代替）
      - pl_top1     : top‑1三連単（例 '1-3-5'）
    を返す。
    """
    s = np.array([row[f"lane{i}_score"] for i in range(1, 7)], dtype=float)
    # 数値安定化（最大値でシフト）
    s = s - np.max(s)
    es = np.exp(s)
    denom0 = float(es.sum())

    best1p, best2p, best1 = -1.0, -1.0, None
    for a, b, c in permutations(range(6), 3):
        d1 = denom0
        d2 = d1 - float(es[a])
        d3 = d2 - float(es[b])
        if d2 <= 0 or d3 <= 0:
            continue
        p = (float(es[a]) / d1) * (float(es[b]) / d2) * (float(es[c]) / d3)
        if p > best1p:
            best2p = best1p
            best1p = float(p)
            best1 = (a, b, c)
        elif p > best2p:
            best2p = float(p)

    return pd.Series({
        "pl_top1_prob": best1p,
        "pl_top2_prob": (best2p if best2p >= 0 else np.nan),
        "pl_gap": (best1p - best2p) if best2p >= 0 else np.nan,
        "pl_top1": (f"{best1[0]+1}-{best1[1]+1}-{best1[2]+1}" if best1 is not None else pd.NA),
    })

# レース単位の要約（race_key, race_date, venue はメタ）
_pl_summary = pred_scores_df.apply(_pl_top1_from_scores_row, axis=1)
pred_race_summary = pd.concat(
    [pred_scores_df[["race_key", "race_date"]], _pl_summary],
    axis=1
)
# 予測用のレース要約CSV（新規）
pred_race_summary.to_csv("artifacts/pred_race_summary.csv", index=False)

# 既存の三連単TOP-K CSV にも列を付与して上書き保存（rank に関わらず同じ値を持つ）
tri_df = tri_df.merge(
    pred_race_summary[["race_key", "pl_top1_prob", "pl_top2_prob", "pl_gap", "pl_top1"]],
    on="race_key",
    how="left"
)
tri_df.to_csv("artifacts/pred_trifecta_topk.csv", index=False)


# In[ ]:


# connのクローズ
conn.close()
print("[predict] Prediction completed and saved to artifacts directory.")


# In[ ]:


torch.save({
    "state_dict": model.state_dict(),
    "scaler": scaler_filename,
    "venue2id": venue2id,
    # "race_type2id": race_type2id
}, "cplnet_checkpoint.pt")

