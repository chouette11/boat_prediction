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
class ROIAnalyzer:
    def __init__(self, model, scaler, num_cols: List[str], device: str = "cpu", batch_size: int = 512):
        self.model = model
        self.scaler = scaler
        self.num_cols = num_cols
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def _preprocess_core(df: pd.DataFrame, scaler, num_cols, *, pred_mode: bool = False) -> pd.DataFrame:
        df = df.copy()

        # (A) directional wind features
        if "wind_dir_deg" in df.columns:
            df["wind_dir_rad"] = np.deg2rad(df["wind_dir_deg"])
            df["wind_sin"] = np.sin(df["wind_dir_rad"])
            df["wind_cos"] = np.cos(df["wind_dir_rad"])
        else:
            msg = "⚠️ (pred) wind_dir_deg が存在しないため wind_sin / wind_cos をスキップします。" if pred_mode \
                  else "⚠️ wind_dir_deg が存在しないため wind_sin / wind_cos をスキップします。"
            print(msg)

        # (B) numeric transform (order‑preserving; align to scaler.fit columns)
        expected = list(num_cols)
        if hasattr(scaler, "mean_") and getattr(scaler, "mean_", None) is not None \
           and len(scaler.mean_) == len(expected):
            for i, col in enumerate(expected):
                if col not in df.columns:
                    # back‑fill missing column with train mean → standardized 0
                    df[col] = float(scaler.mean_[i])
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(float(scaler.mean_[i]))
        else:
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[expected] = scaler.transform(df[expected])

        # (C) boolean flags normalization
        bool_cols = [c for c in df.columns if c.endswith("_fs_flag")]
        if len(bool_cols):
            df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

        # (D) safety columns for ranks (keeps downstream code stable)
        rank_cols = [f"lane{l}_rank" for l in range(1, 7)]
        for c in rank_cols:
            if c not in df.columns:
                df[c] = 7
        df[rank_cols] = df[rank_cols].fillna(7).astype("int32")

        # (E) prediction‑only placeholders for finishers
        if pred_mode:
            for c, v in zip(["first_lane", "second_lane", "third_lane"], [1, 2, 3]):
                if c not in df.columns:
                    df[c] = v
            df[["first_lane", "second_lane", "third_lane"]] = \
                df[["first_lane", "second_lane", "third_lane"]].astype("int32")

        return df

    @staticmethod
    def preprocess_df(df: pd.DataFrame, scaler, num_cols) -> pd.DataFrame:
        return ROIAnalyzer._preprocess_core(df, scaler, num_cols, pred_mode=False)

    def _build_loader(self, df: pd.DataFrame) -> DataLoader:
        ds_eval = BoatRaceDatasetBase(df)
        return DataLoader(ds_eval, batch_size=self.batch_size, shuffle=False)

    def _create_loader(self, df_eval: pd.DataFrame, is_pred: bool = False) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
        df = self.preprocess_df(df_eval, self.scaler, self.num_cols)
        loader = self._build_loader(df)
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
        return ROIAnalyzer._preprocess_core(df, scaler, num_cols, pred_mode=True)

    def _create_loader_pred(self, df_eval: pd.DataFrame) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
        df = self.preprocess_df_pred(df_eval, self.scaler, self.num_cols)
        loader = self._build_loader(df)
        return loader, df, pd.DataFrame()

    def predict_scores(self, df_eval: pd.DataFrame, include_meta: bool = True, save_to: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with lane1_score..lane6_score (logits). Optionally appends meta columns.
        """
        loader, df, _ = self._create_loader_pred(df_eval)
        # Determine expected dims from the underlying DualHeadRanker even if wrapped by _RankOnly
        core = getattr(self.model, "base", self.model)
        exp_ctx = int(core.ctx_fc.in_features)
        exp_boat_linear_in = int(core.boat_fc.in_features)
        lane_dim = int(core.lane_emb.embedding_dim)
        exp_boat = exp_boat_linear_in - lane_dim

        self.model.eval()
        outs = []
        with torch.inference_mode():
            for ctx, boats, lane_ids, _, __, ___ in loader:
                # --- align ctx dims (pad with zeros or truncate tail) ---
                if ctx.size(1) != exp_ctx:
                    if ctx.size(1) < exp_ctx:
                        pad = torch.zeros(ctx.size(0), exp_ctx - ctx.size(1), dtype=ctx.dtype, device=ctx.device)
                        ctx = torch.cat([ctx, pad], dim=1)
                    else:
                        ctx = ctx[:, :exp_ctx]

                # --- align boats dims (B,6,D) to expected per-boat feature count ---
                if boats.size(2) != exp_boat:
                    if boats.size(2) < exp_boat:
                        pad = torch.zeros(boats.size(0), boats.size(1), exp_boat - boats.size(2), dtype=boats.dtype, device=boats.device)
                        boats = torch.cat([boats, pad], dim=2)
                    else:
                        boats = boats[:, :, :exp_boat]

                # --- ensure lane_ids shape is (B,6) ---
                if lane_ids.dim() == 1:
                    lane_ids = lane_ids.unsqueeze(1).expand(-1, boats.size(1))
                elif lane_ids.dim() == 2 and lane_ids.size(1) == 1 and boats.size(1) > 1:
                    lane_ids = lane_ids.expand(-1, boats.size(1))

                ctx = ctx.to(self.device)
                boats = boats.to(self.device)
                lane_ids = lane_ids.to(self.device)

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
            Sr = S[r]
            Sr = Sr - Sr.max()                  # numerical stability
            es = np.exp(tau * Sr)               # (6,)
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