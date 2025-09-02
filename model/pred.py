#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:







# In[23]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

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
from BoatRaceDataset2 import BoatRaceDataset     # ← MTL 対応版
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
    """
    Materialise features declared in the registry.

    Parameters
    ----------
    df : DataFrame
        Source dataframe.
    include / exclude : list[str] | None
        White‑/black‑lists of feature names.  `include=None` means “all”.
    inplace : bool
        If False (default), work on a copy to avoid side‑effects.
    """
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


# In[24]:


import nbformat
from nbconvert import PythonExporter

with open("pred.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

with open("pred.py", "w", encoding="utf-8") as f:
    f.write(source)


# In[25]:


load_dotenv(override=True)

DB_CONF = {
    "host":     os.getenv("PGHOST", "localhost"),
    "port":     int(os.getenv("PGPORT", 5432)),
    "dbname":   os.getenv("PGDATABASE", "boatrace"),
    "user":     os.getenv("PGUSER", "br_user"),
    "password": os.getenv("PGPASSWORD", "secret"),
}

conn = psycopg2.connect(**DB_CONF)
result_df = pd.read_sql("""
    SELECT * FROM feat.train_features3
    WHERE race_date <= '2024-12-31'
""", conn)

print(f"Loaded {len(result_df)} rows from the database.")


# In[26]:


# --- 追加特徴量（Feature Registry 経由） ---
result_df = apply_features(result_df)

exclude = []

for lane in range(1, 7):
      # --- 対象列を決める（ターゲット & キー列は除外） ---
      exclude.append(
            f"lane{lane}_bf_course",
      )
      exclude.append(f"lane{lane}_bf_st_time")

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

# rank_cols = [f"lane{l}_rank" for l in range(1, 7)]
# df[rank_cols] = df[rank_cols].fillna(7).astype("int32")
result_df.to_csv("artifacts/train_features.csv", index=False)
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


# In[27]:


def encode(col):
    uniq = sorted(result_df[col].dropna().unique())
    mapping = {v:i for i,v in enumerate(uniq)}
    result_df[col + "_id"] = result_df[col].map(mapping).fillna(-1).astype("int16")
    return mapping
venue2id = encode("venue")
# race_type2id = encode("race_type")


# In[28]:


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
peek_one(result_df)
# ============================================================


# ---------------- Loss / Regularization Weights -----------------
LAMBDA_ST = 0.1      # weight for ST‑MSE  (was 0.3)
L1_ALPHA  = 0.02     # weight for rank‑L1 loss
CLIP_NORM = 10.0     # gradient‑clipping threshold (was 5.0)
RANKNET_ALPHA = 0.10   # weight for pairwise RankNet loss
TEMPERATURE   = 0.80   # logits are divided by T at inference
LAMBDA_WIN = 1.0        # weight for winner‑BCE loss


# In[29]:


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


# In[30]:


result_df["race_date"] = pd.to_datetime(result_df["race_date"]).dt.date
latest_date = result_df["race_date"].max()
cutoff = latest_date - dt.timedelta(days=90)

mode = "diff"  # "raw", "log", "zscore" も試せる
ds_train = BoatRaceDataset(result_df[result_df["race_date"] <  cutoff])
ds_val   = BoatRaceDataset(result_df[result_df["race_date"] >= cutoff])

loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
loader_val   = DataLoader(ds_val,   batch_size=512)

# ------------------- ⑤ 学習ループ（LR↓ + Clip） --------------
device = "cuda" if torch.cuda.is_available() else "cpu"
boat_dim = ds_train.boat_dim
model = DualHeadRanker(boat_in=boat_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)


# In[31]:


# ---- Monkey‑patch ROIAnalyzer so it uses BoatRaceDataset2 (MTL) ----------
from types import MethodType
from BoatRaceDataset2 import BoatRaceDataset as BR2Dataset
from torch.utils.data import DataLoader


class _EvalDatasetMTL(torch.utils.data.Dataset):
    """
    Wrap BoatRaceDataset2 but return only 4 items (ctx, boats, lane_ids, ranks)
    so that roi_util.py can stay unchanged.
    """
    def __init__(self, df):
        self.ds = BR2Dataset(df)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        ctx, boats, lane_ids, ranks, _, _ = self.ds[idx]
        return ctx, boats, lane_ids, ranks

def _create_loader_mtl(self, df_eval: pd.DataFrame):
    """Replacement for ROIAnalyzer._create_loader (MTL‑aware)."""
    df = self.preprocess_df(df_eval, self.scaler, self.num_cols)
    ds_eval = _EvalDatasetMTL(df)
    loader = DataLoader(ds_eval, batch_size=self.batch_size, shuffle=False)

    need_cols = ["first_lane", "second_lane", "third_lane"]
    if all(c in df.columns for c in need_cols):
        lanes_np = df[need_cols].to_numpy(dtype=np.int64) - 1
    else:
        # 予測テーブルでは真の1〜3着が無いのが普通。ダミー(0,1,2)で形だけ満たす
        lanes_np = np.tile(np.array([0, 1, 2], dtype=np.int64), (len(df), 1))
    return loader, df, lanes_np
    

import roi_util as _roi_util_mod
_roi_util_mod.ROIAnalyzer._create_loader = _create_loader_mtl
from roi_util import ROIAnalyzer


 # 最新のモデルを取得
model_list = os.listdir("artifacts/models")
model_list = [f for f in model_list if f.endswith(".pth")]
if model_list:
    latest_model = sorted(model_list)[-1]  # 最新のモデルを選択
    model_path = os.path.join("artifacts", "models", latest_model)
    print(f"Using latest model: {model_path}")
    # モデルをロード
    model = DualHeadRanker(boat_in=boat_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))

today = dt.date.today()
# 2025年1月1日以降のデータを取得する場合は、以下の行を変更してください。
start_date = dt.date(2025, 1, 1)
# start_date = today - dt.timedelta(days=20)

query = f"""
    SELECT * FROM pred.eval_with_record
    WHERE race_date BETWEEN '{start_date}' AND '{today}'
"""
df_recent = pd.read_sql(query, conn)
df_recent.to_csv("artifacts/eval_features.csv", index=False)
print(df_recent)

df_recent.drop(columns=exclude, inplace=True, errors="ignore")

if df_recent.empty:
    print("[simulate] No rows fetched for last 3 months.")

print(f"[simulate] Loaded {len(df_recent)} rows ({start_date} – {today}).")
print(f"columns: {', '.join(df_recent.columns)}")

# ---- wrap MTL model so ROIAnalyzer sees only rank scores ----
class _RankOnly(nn.Module):
    """Adapter: forward() returns rank_pred tensor only, temperature-scaled."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, *args, **kwargs):
        _, rank_pred, _ = self.base(*args, **kwargs)
        return rank_pred / TEMPERATURE

# ----- metrics & equity (best‑practice defaults) -----
rank_model = _RankOnly(model).to(device)

analyzer = ROIAnalyzer(model=rank_model, scaler=scaler,
                       num_cols=NUM_COLS, device=device)

# df_trifecta_met = analyzer.compute_metrics_dataframe(
#     df_eval=df_recent,
#     tau=5.0,                 # ← Fractional‑Kelly倍率を上げてユニットを実用域へ
#     calibrate="platt",        # ← Platt scaling で確率をキャリブレーション
#     bet_type="trifecta",  # ← 三連単を対象にする
# )

# df_trifecta_met.to_csv("artifacts/metrics_trifecta.csv", index=False)

# # hitが True の行だけを抽出
# df_trifecta_met_hit = df_trifecta_met[df_trifecta_met["hit"] == True]
# df_trifecta_met_hit.to_csv("artifacts/metrics_trifecta_hit.csv", index=False)


# In[32]:


# --- 予測でも「自信度」と「正解三連単の順位」を評価し、CSV に記録 ---
print("[predict] Evaluating confidence & trifecta rank on recent predictions…")

# ROIAnalyzer の前処理（スケーリング等）をそのまま使ってローダを作成
loader_eval, _df_eval_proc, _df_odds = analyzer._create_loader(df_recent)
# df_eval_proc のcolumnsを確認
print(f"[predict] df_eval_proc columns: {', '.join(_df_eval_proc.columns)}")
# ここで _df_eval_proc は、ROIAnalyzer._create_loader() で

# 既に上で用意した rank_model は「rank_pred だけ」を返すアダプタ
model.eval(); rank_model.eval()
# --- prepare lists ---
all_scores, all_ranks, all_keys, all_odds = [], [], [], []

row_ptr = 0
with torch.no_grad():
    for ctx, boats, lane_ids, ranks in loader_eval:
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


# all_ranksとall_scoresを結合したdfに変換
df_scores = pd.DataFrame(all_scores.numpy(), columns=[f"lane{i+1}_score" for i in range(6)])
df_ranks = pd.DataFrame(all_ranks.numpy(), columns=[f"lane{i+1}_rank" for i in range(6)])
df_score_ranks = pd.concat([df_scores, df_ranks], axis=1)   
df_score_ranks["race_key"] = all_keys

# df_mergedから重複行を削除
df_score_ranks = df_score_ranks.drop_duplicates()

# merge odds from df_recent by race_key
df_score_ranks = df_score_ranks.merge(df_recent[["race_key","odds"]], on="race_key", how="left")

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
df_score_ranks.to_csv("artifacts/merged_scores_ranks.csv", index=False)

# df_score_ranksを行でループ
total_benefit = 0.0
total_submit = 0.0
for n in range(1, 6):
    for index, row in df_score_ranks.iterrows():
        total_submit += 100 * n
        odds = row.get("odds", None)
        true_rank = row.get("true_order_rank", None)
        if true_rank <= 1 * n:
            total_benefit += odds * 100

    print(f"n = {n}")
    print(f"roi : {total_benefit / total_submit * 100:.2f}%")



# In[34]:


def run_env_condition_analysis(df_score_ranks, df_recent):
    """環境条件別のHit率/ROI集計を実行し結果CSVを出力する"""
    print("[env] 天候・風・波などの条件別分析を開始…")
    if df_score_ranks is None or df_score_ranks.empty:
        print("[env][error] df_score_ranks が空です。前段の処理を確認してください。")
        return

    base_cols = [c for c in ["race_key", "odds", "true_order_rank"] if c in df_score_ranks.columns]
    if "race_key" not in base_cols:
        print("[env][error] 'race_key' がありません。キー列生成/マージを確認してください。")
        return

    _base = df_score_ranks[base_cols].dropna(subset=["race_key"]).copy()
    _base["true_order_rank"] = pd.to_numeric(_base["true_order_rank"], errors="coerce")
    _base["odds"] = pd.to_numeric(_base["odds"], errors="coerce")
    print(f"[env] base rows after race_key filter: {len(_base)}")

    if df_recent is None or df_recent.empty:
        print("[env][error] df_recent が空です。クエリ取得部分をご確認ください。")
        return

    # merge env columns
    env_cols = ["race_key","weather_txt","wind_speed","wind_dir_deg","wave_height",
                "air_temp","water_temp","venue"]
    _env = df_recent[env_cols].drop_duplicates("race_key")
    _base = _base.merge(_env, on="race_key", how="left")
    print(f"[env] merged env cols: {[c for c in _env.columns if c!='race_key']}")

    # fill sin/cos
    if "wind_dir_deg" in _base.columns:
        _base["wind_sin"] = np.sin(np.deg2rad(_base["wind_dir_deg"]))
        _base["wind_cos"] = np.cos(np.deg2rad(_base["wind_dir_deg"]))

    # bins
    cut = pd.cut
    _base["wind_speed_bin"] = cut(_base["wind_speed"], bins=[-np.inf,2,4,6,8,np.inf])
    _base["wave_height_bin"] = cut(_base["wave_height"], bins=[-np.inf,0.5,1,2,np.inf])
    _base["air_temp_bin"] = pd.qcut(_base["air_temp"], 4, duplicates="drop")
    _base["water_temp_bin"] = pd.qcut(_base["water_temp"],4,duplicates="drop")

    # wind direction discrete labels
    def deg_to_compass8(deg):
        if pd.isna(deg): return np.nan
        d = float(deg)%360.0
        idx = int((d+22.5)//45)%8
        return ["N","NE","E","SE","S","SW","W","NW"][idx]
    _base["wind_compass8"] = _base["wind_dir_deg"].apply(deg_to_compass8)
    def relative_w(sv):
        if pd.isna(sv): return np.nan
        if sv < -0.2: return "tailwind"
        if sv > 0.2:  return "headwind"
        return "cross"
    _base["wind_relative"] = _base["wind_sin"].apply(relative_w)

    # summarize
    def summarize(df, group_cols, top_n, min_n=50):
        d = df.dropna(subset=group_cols+["true_order_rank"]).copy()
        g = d.groupby(group_cols,dropna=False)
        out = g.apply(
            lambda t: pd.Series({
                "n":len(t),
                "hit_rate":(t["true_order_rank"]<=top_n).mean(),
                "avg_odds_on_hits": t.loc[t["true_order_rank"]<=top_n,"odds"].mean(),
                "roi":((t.loc[t['true_order_rank']<=top_n,'odds'].sum() - len(t)*top_n)/(len(t)*top_n))
            })
        ).reset_index()
        out = out[out["n"]>=min_n]
        out["top_n"]=top_n
        out["condition"]= " × ".join(group_cols)
        return out

    single_axes=["weather_txt","wind_speed_bin","wind_compass8","wind_relative",
                 "wave_height_bin","air_temp_bin","water_temp_bin","venue"]
    pair_axes=[["weather_txt","wind_speed_bin"],
               ["weather_txt","wave_height_bin"],
               ["wind_relative","wind_speed_bin"],
               ["venue","wind_relative"]]

    tables=[]
    for N in [1,2,3,4,5]:
        for col in single_axes:
            if col in _base.columns:
                tables.append(summarize(_base,[col],N))
        for cols in pair_axes:
            if set(cols).issubset(_base.columns):
                tables.append(summarize(_base,cols,N))

    env_result = pd.concat(tables,ignore_index=True)
    # save & show
    env_result.to_csv("artifacts/env_cond_hit_roi.csv",index=False)
    print("[env] example Top3 ROI:")
    print(env_result.query("top_n==3").sort_values("roi",ascending=False).head(10))

# --- call the function ---
run_env_condition_analysis(df_score_ranks, df_recent)
# =====================================================================


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
display(exa_df.head())
display(tri_df.head())


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

