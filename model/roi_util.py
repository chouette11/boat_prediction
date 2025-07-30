import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BoatRaceDataset import BoatRaceDataset
from typing import List, Tuple, Dict, Union


class ROIAnalyzer:
    def __init__(self, model, scaler, num_cols: List[str], device: str = "cpu", mode: str = "zscore", batch_size: int = 512):
        self.model = model
        self.scaler = scaler
        self.num_cols = num_cols
        self.device = device
        self.mode = mode
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
        ds_eval = BoatRaceDataset(df, mode=self.mode)
        loader = DataLoader(ds_eval, batch_size=self.batch_size, shuffle=False)
        lanes_np = df[["first_lane", "second_lane", "third_lane"]].to_numpy(dtype=np.int64) - 1
        return loader, df, lanes_np


    def compute_metrics_dataframe(self, df_eval: pd.DataFrame, rake: float = 0.25, tau: float = 1.0) -> pd.DataFrame:
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

        edge = df["odds"].values * np.array(p_model_list) - 1.0
        kelly = (edge / df["odds"].values) * tau  # ← tau をかけて割合を調整

        # --- derive actual (real) finish order for each row --------------------
        # The finishing ranks per lane are stored in lane{n}_rank columns where
        # a lower rank means a better finish position.  We convert these ranks
        # to the lane‑number order array [act1, act2, act3] representing the
        # real 1st‑, 2nd‑, and 3rd‑place lanes.
        rank_cols = [f"lane{l}_rank" for l in range(1, 7) if f"lane{l}_rank" in df.columns]
        if rank_cols:                                               # ranks available
            ranks_arr = df[rank_cols].to_numpy()
            act_order = np.argsort(ranks_arr, axis=1) + 1           # lane numbers (1‑6)
            act1 = act_order[:, 0]
            act2 = act_order[:, 1]
            act3 = act_order[:, 2]
        else:                                                       # safety fallback
            act1 = act2 = act3 = np.full(len(df), np.nan)

        # --- hit/returns should compare the **bet combination** with the **real finish order** ---
        bet_combo = df[["first_lane", "second_lane", "third_lane"]].to_numpy()
        actual_finish = np.column_stack([act1, act2, act3])         # derived above
        hit_mask = (bet_combo == actual_finish).all(axis=1)         # True if bet equals result
        returns = np.where(hit_mask, df["odds"].values, 0.0)        # payoff only when hit

        df_met = df[["race_key", "first_lane", "second_lane", "third_lane", "odds"]].copy()
        df_met[["pred1", "pred2", "pred3"]] = preds
        df_met["conf"] = confs
        df_met["p_model"] = p_model_list
        df_met["edge"] = edge
        df_met["kelly"] = kelly
        df_met["hit"] = hit_mask
        df_met["returns"] = returns
        df_met[["act1", "act2", "act3"]] = np.column_stack([act1, act2, act3])

        return df_met


    @staticmethod
    def compute_equity_curve(df_met: pd.DataFrame,
                             bet_unit: float = 1.0,
                             bet_mode: str = "kelly",
                             *,
                             min_conf: float = None,
                             min_edge: float = None,
                             min_kelly: float = None) -> pd.DataFrame:
        """Compute per‑bet and cumulative PnL (units & JPY).

        Parameters
        ----------
        df_met : pd.DataFrame
            Output of `compute_metrics_dataframe`.
        bet_unit : float, default 1.0
            Wager multiplier when `bet_mode="kelly"` or fixed bet size when `bet_mode="fixed"`.
        bet_mode : {"kelly", "fixed"}
        min_conf : float | None, optional
            Require confidence (margin) >= this threshold.
        min_edge : float | None, optional
            Require edge >= this threshold.
        min_kelly : float | None, optional
            Require Kelly fraction >= this threshold.
        """
        # optional filtering by thresholds
        mask = np.ones(len(df_met), dtype=bool)
        if min_conf is not None and "conf" in df_met.columns:
            mask &= df_met["conf"] >= min_conf
        if min_edge is not None and "edge" in df_met.columns:
            mask &= df_met["edge"] >= min_edge
        if min_kelly is not None and "kelly" in df_met.columns:
            mask &= df_met["kelly"] >= min_kelly

        df = df_met.loc[mask].copy()
        print(f"[compute_equity_curve] Using {len(df)} bets out of {len(df_met)} after thresholding.")

        # 投資量の計算
        if bet_mode == "kelly":
            df["bet_units"] = df["kelly"].clip(lower=0.0) * bet_unit
        elif bet_mode == "fixed":
            df["bet_units"] = bet_unit
        else:
            raise ValueError(f"Unknown bet_mode: {bet_mode}")

        df["pnl"] = (df["returns"] - 1.0) * df["bet_units"]
        df["pnl_jpy"] = df["pnl"] * 100  # 任意: 単位を円に換算
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