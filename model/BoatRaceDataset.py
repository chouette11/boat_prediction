import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class BoatRaceDataset(Dataset):
    """
    - 数値列 → float32, NaN/±inf は 0.0 に置換
    - rank ∈ {1,2,3,4,5,6,…} を
      *重複しない 1〜6 & 最下位以降* に正規化して返す

    ---- boat features ----
        exh_time,
        (st),
        (fs_flag),
        (weight),
        (delta, violation)
    → len = 1–6 depending on available columns
    """

    def __init__(self, frame: pd.DataFrame, mode: str = "diff"):
        self.f = frame.copy()
        self.mode = mode

        # ------- ST 列の有無を判定 -----------------------------------
        self.has_st = f"lane1_st" in self.f.columns
        self.has_bf_st = f"lane1_bf_st_time" in self.f.columns
        self.has_fs      = f"lane1_fs_flag" in self.f.columns
        self.has_weight  = f"lane1_weight"  in self.f.columns
        self.has_course  = f"lane1_bf_course" in self.f.columns

        # ------- 数値列の型揃え & 欠損処理 -----------------------------
        num_cols = self.f.select_dtypes(include=["number", "bool"]).columns
        self.f[num_cols] = (
            self.f[num_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )

        # ------- StandardScaler (z-score モードのみ) ------------------
        if mode == "zscore":
            self.boat_scaler = StandardScaler()
            boat_feats = []
            for lane in range(1, 7):
                cols = [f"lane{lane}_exh_time"]
                if self.has_st:
                    cols.append(f"lane{lane}_st")
                if self.has_bf_st:
                    cols.append(f"lane{lane}_bf_st_time")
                if self.has_weight:
                    cols.append(f"lane{lane}_weight")
                # fs_flag は 0/1 のカテゴリなのでスケーリングしない
                boat_feats.append(self.f[cols].values)
            boat_all = np.stack(boat_feats, axis=1).reshape(-1, len(cols))
            self.boat_scaler.fit(boat_all)

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
        # exh は必須
        self.boat_dim = 1 \
            + int(self.has_st) \
            + int(self.has_bf_st) \
            + int(self.has_fs) \
            + int(self.has_weight) \
            + (2 if self.has_course else 0)   # delta & violation

    # ================================================================

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        r = self.f.iloc[idx]

        # ❶ 環境特徴量 --------------------------------------------------
        ctx = torch.tensor([
            r["wind_speed"], r["wave_height"],
            r["air_temp"],   r["water_temp"],
            r["wind_sin"],   r["wind_cos"]
        ], dtype=torch.float32)

        # ❷ 各艇の元特徴量を収集 ---------------------------------------
        exh_times = [r.get(f"lane{lane}_exh_time", 0.0) for lane in range(1, 7)]
        st_times  = [r.get(f"lane{lane}_st", 0.0)        for lane in range(1, 7)]
        bf_st_times = [r.get(f"lane{lane}_bf_st_time", 0.0) for lane in range(1, 7)]
        fs_flags  = [float(r.get(f"lane{lane}_fs_flag", 0.0)) for lane in range(1, 7)]
        weights   = [r.get(f"lane{lane}_weight", 0.0)    for lane in range(1, 7)]
        raw_ranks = [int(r[f"lane{lane}_rank"]) for lane in range(1, 7)]
        lane_ids  = list(range(6))

        boats = []
        for i in range(6):
            # ---------- choose base numeric features ------------------
            if self.mode == "diff":
                mean_exh = np.mean(exh_times)
                if self.has_st:
                    mean_st  = np.mean(st_times)
                if self.has_bf_st:
                    mean_bf_st = np.mean(bf_st_times)
                if self.has_weight:
                    mean_wt  = np.mean(weights)

                feat = [exh_times[i] - mean_exh]

                if self.has_st:
                    feat.append(st_times[i] - mean_st)
                if self.has_bf_st:
                    feat.append(bf_st_times[i] - mean_bf_st)
                if self.has_fs:
                    feat.append(fs_flags[i])
                if self.has_weight:
                    feat.append(weights[i] - mean_wt)

            elif self.mode == "raw":
                feat = [exh_times[i]]
                if self.has_st:
                    feat.append(st_times[i])
                if self.has_bf_st:
                    feat.append(bf_st_times[i])
                if self.has_fs:
                    feat.append(fs_flags[i])
                if self.has_weight:
                    feat.append(weights[i])

            elif self.mode == "log":
                feat = [np.log1p(exh_times[i])]
                if self.has_st:
                    feat.append(np.log1p(st_times[i]))
                if self.has_bf_st:
                    feat.append(np.log1p(bf_st_times[i]))
                if self.has_fs:
                    feat.append(fs_flags[i])
                if self.has_weight:
                    feat.append(np.log1p(weights[i]))

            elif self.mode == "zscore":
                inp_list = [exh_times[i]]
                if self.has_st:
                    inp_list.append(st_times[i])
                if self.has_bf_st:
                    inp_list.append(bf_st_times[i])
                if self.has_weight:
                    inp_list.append(weights[i])
                inp = np.array([inp_list])
                scaled = self.boat_scaler.transform(inp)[0]
                idx_s = 0
                feat = [scaled[idx_s]]; idx_s += 1
                if self.has_st:
                    feat.append(scaled[idx_s]); idx_s += 1
                if self.has_bf_st:
                    feat.append(scaled[idx_s]); idx_s += 1
                if self.has_fs:
                    feat.append(fs_flags[i])
                if self.has_weight:
                    feat.append(scaled[idx_s])
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

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

        return (
            ctx,
            torch.stack(boats),                    # (6, boat_dim)
            torch.tensor(lane_ids, dtype=torch.int64),
            torch.tensor(new_rank, dtype=torch.int64),
        )