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

        # --- win pattern rates (per-lane) & lane1 defeat pattern rates ---------
        self.pat_names = ["nige","sashi","makuri","makurizashi","nuki","megumare","other"]
        self.lose_names = ["sashi","makuri","makurizashi","nuki","penalty"]
        cols = set(self.f.columns)
        # use gated if available, else raw
        self.use_gated_pat = any(f"lane1_pat_{p}_rate_gated" in cols for p in self.pat_names)
        self.has_pat_rates = any((f"lane1_pat_{p}_rate_gated" in cols) or (f"lane1_pat_{p}_rate" in cols) for p in self.pat_names)
        self.use_gated_lose = any(f"lane1_lose_{p}_rate_gated" in cols for p in self.lose_names)
        self.has_lane1_lose = any((f"lane1_lose_{p}_rate_gated" in cols) or (f"lane1_lose_{p}_rate" in cols) for p in self.lose_names)

        # compressed axes (attack/chaos/entropy/margin) & compat features flags
        self.has_pat_axes = self.has_pat_rates  # same availability as pattern rates
        self.has_compat   = self.has_pat_rates and self.has_lane1_lose

        # --- motor / boat aggregated stats (global posterior用) -------------
        self.has_motor_stats = (f"lane1_motor_starts" in self.f.columns) and (f"lane1_motor_firsts" in self.f.columns)
        self.has_boat_stats  = (f"lane1_boat_starts"  in self.f.columns) and (f"lane1_boat_firsts"  in self.f.columns)

        # Backward-compatible names for logging
        self.has_motor_rates = self.has_motor_stats
        self.has_boat_rates  = self.has_boat_stats

        # Precomputed posterior/gate columns (optional; created by Feature Registry)
        self.has_motor_post_col = f"lane1_motor_post" in self.f.columns
        self.has_boat_post_col  = f"lane1_boat_post"  in self.f.columns
        self.has_motor_gate_col = f"lane1_motor_gate" in self.f.columns
        self.has_boat_gate_col  = f"lane1_boat_gate"  in self.f.columns

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

        # ----- Bayesian prior params (posterior mean ~ 1/6) ---------------
        self.ALPHA_MTR, self.BETA_MTR = 3.0, 15.0
        self.ALPHA_BOT, self.BETA_BOT = 3.0, 15.0

        def _collect_global_mu_sd_from_post(kind: str):
            vals = []
            for l in range(1, 7):
                col = f"lane{l}_{'motor' if kind=='motor' else 'boat'}_post"
                if col in self.f.columns:
                    vals.append(self.f[col].astype("float32").values)
            if not vals:
                return None
            arr = np.concatenate(vals).astype("float32")
            mu = float(arr.mean())
            sd = float(arr.std() if arr.std() > 1e-6 else 1.0)
            return mu, sd

        def _collect_global_mu_sd(kind: str):
            rates = []
            for _, rr in self.f.iterrows():
                for l in range(1, 7):
                    if kind == "motor":
                        s = rr.get(f"lane{l}_motor_starts", 0.0)
                        f_ = rr.get(f"lane{l}_motor_firsts", 0.0)
                        a, b = self.ALPHA_MTR, self.BETA_MTR
                    else:
                        s = rr.get(f"lane{l}_boat_starts", 0.0)
                        f_ = rr.get(f"lane{l}_boat_firsts", 0.0)
                        a, b = self.ALPHA_BOT, self.BETA_BOT
                    rates.append((f_ + a) / (s + a + b))
            arr = np.array(rates, dtype="float32")
            mu = float(arr.mean())
            sd = float(arr.std() if arr.std() > 1e-6 else 1.0)
            return mu, sd

        # compute global μ/σ per kind（この Dataset 範囲＝通常は場ごと）
        if self.has_motor_stats:
            tmp = _collect_global_mu_sd_from_post("motor") if self.has_motor_post_col else None
            if tmp is not None:
                self.mtr_mu, self.mtr_sd = tmp
            else:
                self.mtr_mu, self.mtr_sd = _collect_global_mu_sd("motor")
        if self.has_boat_stats:
            tmp = _collect_global_mu_sd_from_post("boat") if self.has_boat_post_col else None
            if tmp is not None:
                self.bot_mu, self.bot_sd = tmp
            else:
                self.bot_mu, self.bot_sd = _collect_global_mu_sd("boat")

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
        self.boat_dim = 1 \
            + int(self.has_bf_st) \
            + int(self.has_fs) \
            + int(self.has_weight) \
            + (2 if self.has_course else 0) \
            + int(self.has_first_rate) \
            + int(self.has_two_rate) \
            + int(self.has_three_rate) \
            + int(self.has_motor_stats) \
            + int(self.has_boat_stats) \
            + (len(self.pat_names) if self.has_pat_rates else 0) \
            + (len(self.lose_names) if self.has_lane1_lose else 0) \
            + (4 if self.has_pat_axes else 0) \
            + (1 if self.has_compat else 0)

        # ------- Keep only columns actually consumed by Dataset/pipeline -------
        used = set([
            "race_key", "race_date", "venue",
            "wind_speed", "wave_height", "air_temp", "water_temp",
            "wind_sin", "wind_cos",
        ])
        for l in range(1, 7):
            used.add(f"lane{l}_exh_time")
            used.add(f"lane{l}_st")
            used.add(f"lane{l}_rank")
            if self.has_bf_st:
                used.add(f"lane{l}_bf_st_time")
            if self.has_fs:
                used.add(f"lane{l}_fs_flag")
            if self.has_weight:
                used.add(f"lane{l}_weight")
            if self.has_course:
                used.add(f"lane{l}_bf_course")
            if self.has_first_rate:
                used.add(f"lane{l}_first_rate")
            if self.has_two_rate:
                used.add(f"lane{l}_two_rate")
            if self.has_three_rate:
                used.add(f"lane{l}_three_rate")
            # motor/boat stats or posterior + optional gates
            if self.has_motor_stats:
                if self.has_motor_post_col:
                    used.add(f"lane{l}_motor_post")
                else:
                    used.add(f"lane{l}_motor_starts"); used.add(f"lane{l}_motor_firsts")
                if self.has_motor_gate_col:
                    used.add(f"lane{l}_motor_gate")
            if self.has_boat_stats:
                if self.has_boat_post_col:
                    used.add(f"lane{l}_boat_post")
                else:
                    used.add(f"lane{l}_boat_starts"); used.add(f"lane{l}_boat_firsts")
                if self.has_boat_gate_col:
                    used.add(f"lane{l}_boat_gate")
            # per‑lane win pattern rates (prefer gated if present)
            if self.has_pat_rates:
                for p in self.pat_names:
                    col = f"lane{l}_pat_{p}_rate{'_gated' if self.use_gated_pat else ''}"
                    if col in self.f.columns:
                        used.add(col)
                # axes/compat compute need the gate
                used.add(f"lane{l}_pat_gate")
        # lane1 defeat pattern rates + gate
        if self.has_lane1_lose:
            for p in self.lose_names:
                col = f"lane1_lose_{p}_rate{'_gated' if self.use_gated_lose else ''}"
                if col in self.f.columns:
                    used.add(col)
            used.add("lane1_lose_gate")

        self.used_columns = sorted([c for c in used if c in self.f.columns])
        self.f = self.f[self.used_columns].copy()

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

        # --- Global posterior → Z‑score (motor/boat) + gates ---------------
        K_MTR, K_BOT = 50.0, 50.0  # evidence gate strength

        if getattr(self, "has_motor_stats", False):
            if getattr(self, "has_motor_post_col", False):
                m_rate = np.array([r.get(f"lane{lane}_motor_post", 0.0) for lane in range(1, 7)], dtype="float32")
            else:
                m_starts = [r.get(f"lane{lane}_motor_starts", 0.0) for lane in range(1, 7)]
                m_firsts = [r.get(f"lane{lane}_motor_firsts", 0.0) for lane in range(1, 7)]
                m_rate = np.array(
                    [(mf + self.ALPHA_MTR) / (ms + self.ALPHA_MTR + self.BETA_MTR)
                     for mf, ms in zip(m_firsts, m_starts)],
                    dtype="float32"
                )
            m_rel = (m_rate - self.mtr_mu) / self.mtr_sd
            if getattr(self, "has_motor_gate_col", False):
                m_gate = np.array([r.get(f"lane{lane}_motor_gate", 0.0) for lane in range(1, 7)], dtype="float32")
            else:
                m_starts = [r.get(f"lane{lane}_motor_starts", 0.0) for lane in range(1, 7)]
                m_gate = np.array([ms / (ms + K_MTR) for ms in m_starts], dtype="float32")
        else:
            m_rel = np.zeros(6, dtype="float32")
            m_gate = np.zeros(6, dtype="float32")

        if getattr(self, "has_boat_stats", False):
            if getattr(self, "has_boat_post_col", False):
                b_rate = np.array([r.get(f"lane{lane}_boat_post", 0.0) for lane in range(1, 7)], dtype="float32")
            else:
                b_starts = [r.get(f"lane{lane}_boat_starts", 0.0) for lane in range(1, 7)]
                b_firsts = [r.get(f"lane{lane}_boat_firsts", 0.0) for lane in range(1, 7)]
                b_rate = np.array(
                    [(bf + self.ALPHA_BOT) / (bs + self.ALPHA_BOT + self.BETA_BOT)
                     for bf, bs in zip(b_firsts, b_starts)],
                    dtype="float32"
                )
            b_rel = (b_rate - self.bot_mu) / self.bot_sd
            if getattr(self, "has_boat_gate_col", False):
                b_gate = np.array([r.get(f"lane{lane}_boat_gate", 0.0) for lane in range(1, 7)], dtype="float32")
            else:
                b_starts = [r.get(f"lane{lane}_boat_starts", 0.0) for lane in range(1, 7)]
                b_gate = np.array([bs / (bs + K_BOT) for bs in b_starts], dtype="float32")
        else:
            b_rel = np.zeros(6, dtype="float32")
            b_gate = np.zeros(6, dtype="float32")

        boats = []
        for i in range(6):
            lane_num = i + 1
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

            # dynamically weighted global‑relative motor/boat signals
            if getattr(self, "has_motor_stats", False):
                feat.append(float(m_gate[i] * m_rel[i]))
            if getattr(self, "has_boat_stats", False):
                feat.append(float(b_gate[i] * b_rel[i]))

            # --- per-lane win pattern rates (gated preferred) --------------
            if getattr(self, "has_pat_rates", False):
                for p in self.pat_names:
                    col = f"lane{lane_num}_pat_{p}_rate{'_gated' if self.use_gated_pat else ''}"
                    feat.append(float(r.get(col, 0.0)))

            # --- compressed axes from pattern distribution -----------------
            if getattr(self, "has_pat_axes", False):
                # pull needed rates (gated preferred)
                def _p(name: str) -> float:
                    return float(r.get(f"lane{lane_num}_pat_{name}_rate{'_gated' if self.use_gated_pat else ''}", 0.0))
                p_nige = _p("nige"); p_sashi = _p("sashi"); p_makuri = _p("makuri");
                p_mkz  = _p("makurizashi"); p_nuki  = _p("nuki"); p_meg  = _p("megumare")
                eps = 1e-3
                # axes (log‑ratios) — gate with lane‑level evidence if available
                gate_lane = float(r.get(f"lane{lane_num}_pat_gate", 1.0))
                attack_axis = np.log((p_makuri + 0.7*p_mkz + eps) / (p_nige + p_sashi + 0.3*p_mkz + eps)) * gate_lane
                chaos_axis  = np.log((p_nuki + p_meg + eps) / (p_nige + p_sashi + p_makuri + p_mkz + eps)) * gate_lane
                # entropy over 6 (exclude 'other')
                arr6 = np.array([p_nige, p_sashi, p_makuri, p_mkz, p_nuki, p_meg], dtype=np.float32)
                # add small epsilon to avoid log(0)
                arr6_safe = arr6 + eps
                ent = float(-(arr6_safe * np.log(arr6_safe)).sum() / np.log(6.0)) * gate_lane
                # margin (top1 - top2) over 6
                top2 = np.sort(arr6)[-2:]
                margin = float(top2[-1] - top2[-2]) * gate_lane
                feat.extend([float(attack_axis), float(chaos_axis), float(ent), float(margin)])

            # --- compatibility: attacker lane (2..6) vs lane1 vulnerabilities ---
            if getattr(self, "has_compat", False):
                if lane_num == 1:
                    feat.append(0.0)
                else:
                    # attacker components (sashi/makuri/makurizashi/nuki)
                    att = np.array([
                        float(r.get(f"lane{lane_num}_pat_sashi_rate{'_gated' if self.use_gated_pat else ''}", 0.0)),
                        float(r.get(f"lane{lane_num}_pat_makuri_rate{'_gated' if self.use_gated_pat else ''}", 0.0)),
                        float(r.get(f"lane{lane_num}_pat_makurizashi_rate{'_gated' if self.use_gated_pat else ''}", 0.0)),
                        float(r.get(f"lane{lane_num}_pat_nuki_rate{'_gated' if self.use_gated_pat else ''}", 0.0)),
                    ], dtype=np.float32)
                    # lane1 vulnerability components (same ordering)
                    def _lose(n: str) -> float:
                        return float(r.get(f"lane1_lose_{n}_rate{'_gated' if self.use_gated_lose else ''}", 0.0))
                    vuln = np.array([_lose("sashi"), _lose("makuri"), _lose("makurizashi"), _lose("nuki")], dtype=np.float32)
                    # gate with both attacker evidence and lane1 defeat evidence
                    g_att = float(r.get(f"lane{lane_num}_pat_gate", 1.0))
                    g_def = float(r.get("lane1_lose_gate", 1.0))
                    g = float(np.sqrt(max(g_att, 0.0) * max(g_def, 0.0)))
                    compat = float((att * vuln).sum()) * g
                    feat.append(compat)

            # --- lane1 defeat pattern rates (attach to lane1, zeros otherwise)
            if getattr(self, "has_lane1_lose", False):
                for p in self.lose_names:
                    if lane_num == 1:
                        col = f"lane1_lose_{p}_rate{'_gated' if self.use_gated_lose else ''}"
                        feat.append(float(r.get(col, 0.0)))
                    else:
                        feat.append(0.0)

            # ----- add delta (lane − course) & violation --------------
            if self.has_course:
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