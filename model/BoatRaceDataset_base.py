import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class BoatRaceDatasetBase(Dataset):
    """
    - 数値列 → float32, NaN/±inf は 0.0 に置換
    - rank ∈ {1,2,3,4,5,6,…} を
      *重複しない 1〜6 & 最下位以降* に正規化して返す

    ---- boat features ----
        exh_time,
        (fs_flag),
        (weight),
        (delta, violation),
        (first_rate, two_rate, three_rate)
    → len = 1–9 depending on available columns

    展示：相対差分 / 重量・環境：Z-score / 風向：raw
    """

    def __init__(self, frame: pd.DataFrame):
        self.f = frame.copy()

        # ------- ST 列の有無を判定 -----------------------------------
        self.has_st = f"lane1_st" in self.f.columns
        self.has_bf_st = f"lane1_bf_st_time" in self.f.columns
        self.has_fs      = f"lane1_fs_flag" in self.f.columns
        self.has_weight  = f"lane1_weight"  in self.f.columns
        self.has_course  = f"lane1_bf_course" in self.f.columns
        # --- per‑lane win‑rate columns ---------------------------------
        self.has_first_rate  = f"lane1_first_rate"  in self.f.columns
        self.has_two_rate    = f"lane1_two_rate"    in self.f.columns
        self.has_three_rate  = f"lane1_three_rate"  in self.f.columns

        # ------- 数値列の型揃え & 欠損処理 -----------------------------
        num_cols = self.f.select_dtypes(include=["number", "bool"]).columns
        self.f[num_cols] = (
            self.f[num_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )

        # ------- Global stats for Z‑score features ------------------
        env_cols = ["wind_speed", "wave_height", "air_temp", "water_temp"]
        self.env_mean = self.f[env_cols].mean().values.astype("float32")
        self.env_std  = self.f[env_cols].std(ddof=0).replace(0,1e-6).values.astype("float32")

        if self.has_weight:
            all_w = self.f[[f"lane{l}_weight" for l in range(1,7)]].values.flatten()
            self.weight_mu = float(np.mean(all_w))
            self.weight_sd = float(np.std(all_w) if np.std(all_w) > 1e-6 else 1.0)

        # ------- rank を int64 で保持 (欠損→99) -----------------------
        for lane in range(1, 7):
            col = f"lane{lane}_rank"
            if col in self.f.columns:
                self.f[col] = (
                    self.f[col]
                    .fillna(99)
                    .astype("int64")
                )

        # ------- boat feature dimension ------------------------------
        # exh (1) + bf_st (1) + fs_flag (1) + weight (1) + delta & violation (2) + win rates (3)
        self.boat_dim = 1 \
            + int(self.has_bf_st) \
            + int(self.has_fs) \
            + int(self.has_weight) \
            + (2 if self.has_course else 0) \
            + int(self.has_first_rate) \
            + int(self.has_two_rate) \
            + int(self.has_three_rate)

    # ================================================================

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        r = self.f.iloc[idx]

        # ❶ 環境特徴量 --------------------------------------------------
        env_raw = np.array([
            r["wind_speed"], r["wave_height"],
            r["air_temp"],   r["water_temp"]
        ], dtype="float32")
        env_z = (env_raw - self.env_mean) / self.env_std
        ctx = torch.tensor(np.concatenate([env_z,
                                           [r["wind_sin"], r["wind_cos"]]]),
                           dtype=torch.float32)

        # ❷ 各艇の元特徴量を収集 ---------------------------------------
        exh_times = [r.get(f"lane{lane}_exh_time", 0.0) for lane in range(1, 7)]
        st_times  = [r.get(f"lane{lane}_st", 0.0)        for lane in range(1, 7)]
        bf_st_times = [r.get(f"lane{lane}_bf_st_time", 0.0) for lane in range(1, 7)]
        fs_flags  = [float(r.get(f"lane{lane}_fs_flag", 0.0)) for lane in range(1, 7)]
        weights   = [r.get(f"lane{lane}_weight", 0.0)    for lane in range(1, 7)]
        first_rates  = [r.get(f"lane{lane}_first_rate", 0.0)  for lane in range(1, 7)]
        two_rates    = [r.get(f"lane{lane}_two_rate", 0.0)    for lane in range(1, 7)]
        three_rates  = [r.get(f"lane{lane}_three_rate", 0.0)  for lane in range(1, 7)]
        raw_ranks = [int(r[f"lane{lane}_rank"]) for lane in range(1, 7)]
        lane_ids  = list(range(6))

        boats = []
        for i in range(6):
            mean_exh = np.mean(exh_times)
            if self.has_bf_st:
                mean_bf_st = np.mean(bf_st_times)
            if self.has_weight:
                mean_wt = np.mean(weights)

            # --- core relative / zscore features --------------------
            feat = [exh_times[i] - mean_exh]

            if self.has_bf_st:
                feat.append(bf_st_times[i] - mean_bf_st)

            if self.has_fs:
                feat.append(fs_flags[i])

            if self.has_weight:
                feat.append((weights[i] - self.weight_mu) / self.weight_sd)
            # --- per‑lane win rates (raw 0‑1) ---------------------------
            if self.has_first_rate:
                feat.append(first_rates[i])
            if self.has_two_rate:
                feat.append(two_rates[i])
            if self.has_three_rate:
                feat.append(three_rates[i])

            # ----- add delta (lane − course) & violation --------------
            if self.has_course:
                lane_num  = i + 1
                course    = int(r.get(f"lane{lane_num}_bf_course", lane_num))
                delta     = (lane_num - course) / 5.0
                violation = float(delta != 0.0)
                feat.extend([delta, violation])

            boats.append(torch.tensor(feat, dtype=torch.float32))

        # ---------- rank を一意に付け直す -----------------------------
        order = np.argsort(raw_ranks)
        new_rank = [0] * 6
        for new_pos, lane_idx in enumerate(order, start=1):
            new_rank[lane_idx] = new_pos

        # ----- ground‑truth ST (for multi‑task loss) ------------------
        st_vals  = torch.tensor(st_times, dtype=torch.float32)
        st_mask  = (st_vals != 0.0)          # 0.0 を欠損扱い

        return (
            ctx,
            torch.stack(boats),                         # (6, boat_dim)
            torch.tensor(lane_ids, dtype=torch.int64),  # (6,)
            torch.tensor(new_rank, dtype=torch.int64),  # (6,)
            st_vals,                                    # (6,)
            st_mask,                                    # (6,)
        )