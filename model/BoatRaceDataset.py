import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

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
            r["wind_sin"],   r["wind_cos"]
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