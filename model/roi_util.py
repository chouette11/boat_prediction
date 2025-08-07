import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BoatRaceDataset import BoatRaceDataset
from typing import List, Tuple, Dict, Union, Optional
# --- third‑party for probability calibration ------------------------------
from sklearn.linear_model import LogisticRegression      # Platt scaling
from sklearn.isotonic import IsotonicRegression          # Isotonic calibration

# --- constants ------------------------------------------------------------
UNIT_JPY = 100             # 1 "bet unit" = 100 JPY (競艇の最低賭け金)


class ROIAnalyzer:
    def __init__(self, model, scaler, num_cols: List[str], device: str = "cpu", batch_size: int = 512):
        self.model = model
        self.scaler = scaler
        self.num_cols = num_cols
        self.device = device
        self.batch_size = batch_size
        self.batch_size = batch_size

    @staticmethod
    def preprocess_df(df: pd.DataFrame, scaler, num_cols) -> pd.DataFrame:
        df = df.copy()

        if "wind_dir_deg" in df.columns:
            df["wind_dir_rad"] = np.deg2rad(df["wind_dir_deg"])
            df["wind_sin"] = np.sin(df["wind_dir_rad"])
            df["wind_cos"] = np.cos(df["wind_dir_rad"])
        else:
            print("⚠️ wind_dir_deg が存在しないため wind_sin / wind_cos をスキップします。")

        available_cols = [col for col in num_cols if col in df.columns]
        df[available_cols] = scaler.transform(df[available_cols])

        bool_cols = [c for c in df.columns if c.endswith("_fs_flag")]
        df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

        rank_cols = [f"lane{l}_rank" for l in range(1, 7)]
        df[rank_cols] = df[rank_cols].fillna(7).astype("int32")
        return df

    def _create_loader(self, df_eval: pd.DataFrame) -> Tuple[DataLoader, pd.DataFrame, np.ndarray]:
        df = self.preprocess_df(df_eval, self.scaler, self.num_cols)
        ds_eval = BoatRaceDataset(df)
        loader = DataLoader(ds_eval, batch_size=self.batch_size, shuffle=False)
        lanes_np = df[["first_lane", "second_lane", "third_lane"]].to_numpy(dtype=np.int64) - 1
        return loader, df, lanes_np


    def compute_metrics_dataframe(
        self,
        df_eval: pd.DataFrame,
        tau: float = 1.0,
        calibrate: Optional[str] = None,   # "platt", "isotonic", or None
        bet_type: str = "trifecta"         # "trifecta", "trio", "win_fixed"
    ) -> pd.DataFrame:
        """
        Compute metrics for each bet, including hit/miss, edge, kelly, etc.
        bet_type: "trifecta" (order-sensitive), "trio" (order-agnostic), or "win_fixed" (first_lane must win).
        """
        loader, df, lanes_np = self._create_loader(df_eval)
        print(f"[compute_metrics_dataframe] Preprocessed {len(df)} rows.")

        preds_all, conf_all, p_model_list = [], [], []
        self.model.eval(); row_idx = 0
        with torch.no_grad():
            for ctx, boats, lane_ids, _ in loader:
                ctx, boats, lane_ids = ctx.to(self.device), boats.to(self.device), lane_ids.to(self.device)
                scores = self.model(ctx, boats, lane_ids)
                B = scores.size(0)

                top3 = scores.argsort(dim=1, descending=True)[:, :3] + 1
                preds_all.append(top3.cpu().numpy())

                top2_vals = scores.topk(2, dim=1).values
                conf_batch = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy()
                conf_all.append(conf_batch)

                batch_lanes = torch.from_numpy(lanes_np[row_idx: row_idx + B]).to(self.device)
                probs = torch.softmax(scores, dim=1)
                p_mod_batch = torch.gather(probs, 1, batch_lanes).prod(dim=1).cpu().numpy()
                p_model_list.extend(p_mod_batch.tolist())

                row_idx += B

        preds = np.vstack(preds_all)
        confs = np.concatenate(conf_all)

        rank_cols = [f"lane{l}_rank" for l in range(1, 7) if f"lane{l}_rank" in df.columns]
        if rank_cols:
            ranks_arr = df[rank_cols].to_numpy()
            act_order = np.argsort(ranks_arr, axis=1) + 1
            act1, act2, act3 = act_order[:, 0], act_order[:, 1], act_order[:, 2]
        else:  # safety fallback
            act1 = act2 = act3 = np.full(len(df), np.nan)

        bet_combo = df[["first_lane", "second_lane", "third_lane"]].to_numpy()
        actual_finish = np.column_stack([act1, act2, act3])

        if bet_type == "trifecta":
            # order‑sensitive
            hit_mask = (bet_combo == actual_finish).all(axis=1)

        elif bet_type == "trio":
            # order‑agnostic 3連複
            hit_mask = np.array([
                set(bet_combo[i]) == set(actual_finish[i])
                for i in range(len(df))
            ], dtype=bool)

        elif bet_type == "win_fixed":
            # 1着固定流し（first_lane が 1着になれば的中）
            hit_mask = (act1 == df["first_lane"].to_numpy())

        else:
            raise ValueError(f"Unknown bet_type: {bet_type}")

        p_raw = np.array(p_model_list, dtype=float)

        if calibrate == "platt":
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(p_raw.reshape(-1, 1), hit_mask.astype(int))
            p_calib = lr.predict_proba(p_raw.reshape(-1, 1))[:, 1]

        elif calibrate == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p_raw, hit_mask.astype(int))
            p_calib = ir.transform(p_raw)

        else:  # no calibration
            p_calib = p_raw

        edge = df["odds"].values * p_calib - 1.0
        kelly = (edge / df["odds"].values) * tau

        returns = np.where(hit_mask, df["odds"].values, 0.0)        # payoff only when hit

        df_met = df[["race_key", "first_lane", "second_lane", "third_lane", "odds"]].copy()
        df_met[["pred1", "pred2", "pred3"]] = preds
        df_met["conf"] = confs
        df_met["p_model_raw"] = p_raw
        df_met["p_model"] = p_calib
        df_met["edge"] = edge
        df_met["kelly"] = kelly
        df_met["hit"] = hit_mask
        df_met["returns"] = returns
        df_met[["act1", "act2", "act3"]] = np.column_stack([act1, act2, act3])

        return df_met


    @staticmethod
    def compute_equity_curve(
        df_met: pd.DataFrame,
        bet_unit: float = 1.0,
        bet_mode: str = "kelly",
        *,
        min_conf: Optional[float] = None,
        min_edge: Optional[float] = None,
        min_kelly: Optional[float] = None,
        min_units: float = 1.0,
        max_units: Optional[float] = None,
        round_units: bool = True,
        use_conf_factor: bool = False,
    ) -> pd.DataFrame:
        # ---- (1) optional filtering -------------------------------------------------
        mask = np.ones(len(df_met), dtype=bool)
        if min_conf is not None and "conf" in df_met.columns:
            mask &= df_met["conf"] >= min_conf
        if min_edge is not None and "edge" in df_met.columns:
            mask &= df_met["edge"] >= min_edge
        if min_kelly is not None and "kelly" in df_met.columns:
            mask &= df_met["kelly"] >= min_kelly

        df = df_met.loc[mask].copy()
        print(f"[compute_equity_curve] Using {len(df)} bets out of "
              f"{len(df_met)} after thresholding.")
        if df.empty:
            # フィルタで 0 行になったら即リターン
            return pd.DataFrame()

        # ---- (2) bet sizing ---------------------------------------------------------
        if bet_mode == "kelly":
            # base Kelly sizing (clip at 0 so that negative edge → no bet)
            kelly_units = df["kelly"].clip(lower=0.0) * bet_unit

            # optional: down‑weight by confidence (0–1)
            if use_conf_factor and "conf" in df.columns:
                c = df["conf"].to_numpy(dtype=float)
                if c.size <= 1:
                    c_norm = np.ones_like(c)
                else:
                    c_norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
                kelly_units *= c_norm

            # apply min / max caps only where Kelly > 0
            positive = kelly_units > 0.0
            if min_units is not None:
                kelly_units = np.where(positive,
                                       np.maximum(kelly_units, min_units),
                                       0.0)
            if max_units is not None:
                kelly_units = np.where(positive,
                                       np.minimum(kelly_units, max_units),
                                       0.0)

            # final optional rounding (practical: whole "tickets")
            if round_units:
                kelly_units = np.where(positive,
                                       np.ceil(kelly_units),
                                       0.0)

            df["bet_units"] = kelly_units

        elif bet_mode == "fixed":
            df["bet_units"] = bet_unit

        else:
            raise ValueError(f"Unknown bet_mode: {bet_mode}")

        # ---- (3) PnL ---------------------------------------------------------------
        df["pnl"] = (df["returns"] - 1.0) * df["bet_units"]
        df["pnl_jpy"] = df["pnl"] * UNIT_JPY   # 1 unit → ¥100 換算
        df["cum_pnl"] = df["pnl"].cumsum()
        df["cum_pnl_jpy"] = df["pnl_jpy"].cumsum()
        return df
    
    @staticmethod
    def plot_equity_curve(df_eq: pd.DataFrame,
                          title: str = "Equity Curve",
                          use_jpy: bool = False,
                          group_by: str = "bets",
                          figsize=(8, 4)) -> None:
        """Plot cumulative PnL curve (in units or JPY)."""
        plt.figure(figsize=figsize)

        if group_by == "races" and {"race_date", "race_no"}.issubset(df_eq.columns):
            value_col = "pnl_jpy" if use_jpy else "pnl"
            df_plot = (
                df_eq.groupby(["race_date", "race_no"], as_index=False)[value_col]
                     .sum()
                     .sort_values(["race_date", "race_no"])
            )
            series = df_plot[value_col].cumsum().values
            xlabel = "Races (chronological)"
        else:
            series = (df_eq["cum_pnl_jpy"] if use_jpy else df_eq["cum_pnl"]).values
            xlabel = "Bets (chronological)"

        plt.plot(series)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel(xlabel)
        ylabel = "Cumulative PnL (JPY)" if use_jpy else "Cumulative PnL (units)"
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()