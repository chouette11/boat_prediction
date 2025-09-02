import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BoatRaceDataset_base import BoatRaceDatasetBase
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

    def _create_loader(self, df_eval: pd.DataFrame, is_pred: bool = False) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
        df = self.preprocess_df(df_eval, self.scaler, self.num_cols)
        ds_eval = BoatRaceDatasetBase(df)
        loader = DataLoader(ds_eval, batch_size=self.batch_size, shuffle=False)
        if not is_pred:
            df_odds = df[["race_key", "trifecta_odds"]].copy()
            return loader, df, df_odds
        else:
            return loader, df, pd.DataFrame()

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
        return self._create_loader(df_eval, is_pred=True)

    def predict_scores(self, df_eval: pd.DataFrame, include_meta: bool = True, save_to: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with lane1_score..lane6_score (logits). Optionally appends meta columns.
        """
        loader, df, _ = self._create_loader_pred(df_eval)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for ctx, boats, lane_ids, _, __, ___ in loader:
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