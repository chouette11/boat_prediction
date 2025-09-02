import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BoatRaceDataset import BoatRaceDataset
from typing import List, Tuple, Dict, Union, Optional
# --- third‑party for probability calibration ------------------------------
from sklearn.linear_model import LogisticRegression      # Platt scaling
from sklearn.isotonic import IsotonicRegression          # Isotonic calibration

from itertools import permutations


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
        # Ensure rank columns exist (prediction tables may not have them)
        for c in rank_cols:
            if c not in df.columns:
                df[c] = 7
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

        returns = np.where(hit_mask, df["trifecta_odds"].values, 0.0)        # payoff only when hit

        df_met = df[["race_key", "first_lane", "second_lane", "third_lane", "trifecta_odds"]].copy()
        df_met[["pred1", "pred2", "pred3"]] = preds
        df_met["conf"] = confs
        df_met["p_model_raw"] = p_raw
        df_met["hit"] = hit_mask
        df_met["returns"] = returns
        df_met[["act1", "act2", "act3"]] = np.column_stack([act1, act2, act3])

        return df_met
    
    
    @staticmethod
    def compute_metrics_pl(
            all_scores : List[torch.Tensor],
            all_ranks : List[torch.Tensor],
            df_met_hit : pd.DataFrame,
    ):
        new_df = pd.DataFrame(columns=["race_index", "true_order", "match_index"])
        for i in range(0, len(all_scores), 120):  # 129〜133 レースを表示
            sc   = all_scores[i]
            true = all_ranks[i]

            ordered_true = sorted(range(6), key=lambda k: true[k].item())[:3]
            print(f"\nRace {i}")
            print(" true order :", ordered_true)

            # PL順で上位5件
            perms = list(permutations(range(6), 3))
            es = torch.exp(sc)
            denom0 = es.sum().item()
            perm_probs = []
            for p in perms:
                d2 = denom0 - es[p[0]].item()
                d3 = d2     - es[p[1]].item()
                prob = (es[p[0]] / denom0) * (es[p[1]] / d2) * (es[p[2]] / d3)
                perm_probs.append((p, prob.item()))
            perm_probs.sort(key=lambda x: x[1], reverse=True)
            print(" top-5 by PL :", perm_probs[:5])

            # 正解三連単が何番目に出たか
            ordered_true = [x + 1 for x in ordered_true]
            true_order_tuple = tuple(ordered_true)
            match_index = next((j for j, (order, _) in enumerate(perm_probs) if order == true_order_tuple), -1)
            print(f'matched: {match_index + 1}')

            # new_dfに追加
            new_df.loc[len(new_df)] = {
                "race_index": i,
                "true_order": true_order_tuple,
                "match_index": match_index + 1  # 1-indexed にして表示
            }

        new_df.to_csv("artifacts/new_df.csv", index=False)
        

        # df_met_hitを行でループ
        df_hit = pd.DataFrame(columns=["race_key", "trifecta_odds", "true_order"])
        for i in range(len(df_met_hit)):
            race_key = df_met_hit.iloc[i]["race_key"]
            odds = df_met_hit.iloc[i]["trifecta_odds"]
            # np.int64 から int に変換
            first_lane = int(df_met_hit.iloc[i]["first_lane"])
            second_lane = int(df_met_hit.iloc[i]["second_lane"])
            third_lane = int(df_met_hit.iloc[i]["third_lane"])
            true_order = (first_lane, second_lane, third_lane)

            df_hit.loc[len(df_hit)] = {
                "race_key": race_key,
                "trifecta_odds": odds,
                "true_order": true_order
            }

        df_hit.to_csv("artifacts/df_hit.csv", index=False)

        # new_df と df_hit をtrue_orderで結合
        merged_df = pd.merge(new_df, df_hit, left_on="true_order", right_on="true_order", how="inner")
        return merged_df

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


# ===== Prediction-time scorer: produce scores & probabilities without labels =====
class ROIPredictor(ROIAnalyzer):
    """
    Lightweight helper for inference-time outputs.
    - Robust to missing rank / finish columns (fills safe defaults)
    - Returns lane scores (logits) and win probabilities (softmax)
    """
    @staticmethod
    def preprocess_df_pred(df: pd.DataFrame, scaler, num_cols) -> pd.DataFrame:
        df = df.copy()

        # (1) directional wind features if available
        if "wind_dir_deg" in df.columns:
            df["wind_dir_rad"] = np.deg2rad(df["wind_dir_deg"])
            df["wind_sin"] = np.sin(df["wind_dir_rad"])
            df["wind_cos"] = np.cos(df["wind_dir_rad"])
        else:
            print("⚠️ (pred) wind_dir_deg が存在しないため wind_sin / wind_cos をスキップします。")

        # (2) scale numeric columns that are present
        available_cols = [col for col in num_cols if col in df.columns]
        if len(available_cols):
            df[available_cols] = scaler.transform(df[available_cols])

        # (3) bool flags
        bool_cols = [c for c in df.columns if c.endswith("_fs_flag")]
        if len(bool_cols):
            df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

        # (4) ensure lane*_rank exist (not used at pred-time but keeps downstream code safe)
        rank_cols = [f"lane{l}_rank" for l in range(1, 7)]
        for c in rank_cols:
            if c not in df.columns:
                df[c] = 7
        df[rank_cols] = df[rank_cols].fillna(7).astype("int32")

        # (5) ensure placeholders for first/second/third finishers (not used for probs)
        for c, v in zip(["first_lane", "second_lane", "third_lane"], [1, 2, 3]):
            if c not in df.columns:
                df[c] = v
        df[["first_lane", "second_lane", "third_lane"]] = df[["first_lane", "second_lane", "third_lane"]].astype("int32")

        return df

    def _create_loader_pred(self, df_eval: pd.DataFrame) -> Tuple[DataLoader, pd.DataFrame, np.ndarray]:
        # Use the same preprocessing & loader as ROIAnalyzer (unified path / BR2Dataset via monkey-patch)
        return self._create_loader(df_eval)

    def predict_scores(self, df_eval: pd.DataFrame, include_meta: bool = True, save_to: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with lane1_score..lane6_score (logits). Optionally appends meta columns.
        """
        loader, df, _ = self._create_loader_pred(df_eval)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for ctx, boats, lane_ids, _ in loader:
                ctx, boats, lane_ids = ctx.to(self.device), boats.to(self.device), lane_ids.to(self.device)
                scores = self.model(ctx, boats, lane_ids)  # (B,6) logits
                outs.append(scores.cpu())
        scores = torch.cat(outs, dim=0) if len(outs) else torch.empty((0, 6))
        pred_df = pd.DataFrame(scores.numpy(), columns=[f"lane{i}_score" for i in range(1, 7)])

        if include_meta:
            for m in ["race_key", "race_date", "venue"]:
                if m in df.columns:
                    pred_df[m] = df[m].values[: len(pred_df)]

        if save_to:
            pred_df.to_csv(save_to, index=False)
        return pred_df

    def predict_win_probs(self, df_eval: Optional[pd.DataFrame] = None, scores_df: Optional[pd.DataFrame] = None,
                          include_meta: bool = True, save_to: Optional[str] = None) -> pd.DataFrame:
        """
        Returns lane-wise win probabilities and fair odds in *wide* format.
        If scores_df is not provided, it will compute scores from df_eval.
        """
        if scores_df is None:
            if df_eval is None:
                raise ValueError("Either df_eval or scores_df must be provided.")
            scores_df = self.predict_scores(df_eval, include_meta=include_meta)

        score_cols = [f"lane{i}_score" for i in range(1, 7)]
        S = scores_df[score_cols].to_numpy()
        # numerically stable softmax
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        P = E / E.sum(axis=1, keepdims=True)

        prob_df = pd.DataFrame(P, columns=[f"lane{i}_prob" for i in range(1, 7)])
        fair_df = pd.DataFrame(1.0 / np.clip(P, 1e-12, None), columns=[f"lane{i}_fair_odds" for i in range(1, 7)])

        meta = scores_df.drop(columns=score_cols, errors="ignore") if include_meta else pd.DataFrame(index=scores_df.index)
        out = pd.concat([meta.reset_index(drop=True), prob_df, fair_df], axis=1)

        if save_to:
            out.to_csv(save_to, index=False)
        return out
    
    def predict_exotics_topk(self,
                            df_eval: Optional[pd.DataFrame] = None,
                            scores_df: Optional[pd.DataFrame] = None,
                            K: int = 10,
                            tau: float = 5.0,
                            include_meta: bool = True,
                            save_exacta: Optional[str] = None,
                            save_trifecta: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute PL‑style probabilities for Exacta (1‑2) and Trifecta (1‑2‑3),
        return top‑K candidates per race, with fair odds (1/p).
        Either provide `scores_df` from predict_scores() or a raw `df_eval`.
        """
        if scores_df is None:
            if df_eval is None:
                raise ValueError("Either df_eval or scores_df must be provided.")
            scores_df = self.predict_scores(df_eval, include_meta=include_meta)

        score_cols = [f"lane{i}_score" for i in range(1, 7)]
        S = scores_df[score_cols].to_numpy(dtype=float)  # (N,6)
        N = S.shape[0]
        meta_cols = [c for c in ["race_key", "race_date", "venue"] if c in scores_df.columns] if include_meta else []

        exacta_rows, trifecta_rows = [], []
        for r in range(N):
            es = np.exp(tau * S[r])             # (6,)
            denom0 = es.sum()
            # --- Exacta ---
            pairs = []
            for i in range(6):
                for j in range(6):
                    if j == i:
                        continue
                    p_e = (es[i] / denom0) * (es[j] / (denom0 - es[i]))
                    pairs.append(((i + 1, j + 1), float(p_e)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            topE = pairs[: min(K, len(pairs))]
            for rank, (pair, p) in enumerate(topE, start=1):
                row = {
                    "rank": rank,
                    "exacta": f"{pair[0]}-{pair[1]}",
                    "prob": p,
                    "fair_odds": float(1.0 / p) if p > 0 else np.inf,
                }
                for m in meta_cols:
                    row[m] = scores_df.iloc[r][m]
                exacta_rows.append(row)

            # --- Trifecta ---
            trips = []
            for i in range(6):
                for j in range(6):
                    if j == i:
                        continue
                    d2 = denom0 - es[i]
                    for k in range(6):
                        if k == i or k == j:
                            continue
                        d3 = d2 - es[j]
                        p_t = (es[i] / denom0) * (es[j] / d2) * (es[k] / d3)
                        trips.append(((i + 1, j + 1, k + 1), float(p_t)))
            trips.sort(key=lambda x: x[1], reverse=True)
            topT = trips[: min(K, len(trips))]
            for rank, (trip, p) in enumerate(topT, start=1):
                row = {
                    "rank": rank,
                    "trifecta": f"{trip[0]}-{trip[1]}-{trip[2]}",
                    "prob": p,
                    "fair_odds": float(1.0 / p) if p > 0 else np.inf,
                }
                for m in meta_cols:
                    row[m] = scores_df.iloc[r][m]
                trifecta_rows.append(row)

        df_exacta = pd.DataFrame(exacta_rows, columns=(meta_cols + ["rank", "exacta", "prob", "fair_odds"]))
        df_trifecta = pd.DataFrame(trifecta_rows, columns=(meta_cols + ["rank", "trifecta", "prob", "fair_odds"]))

        if save_exacta:
            df_exacta.to_csv(save_exacta, index=False)
        if save_trifecta:
            df_trifecta.to_csv(save_trifecta, index=False)
        return df_exacta, df_trifecta