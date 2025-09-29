from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app, firestore
from firebase_functions.options import MemoryOption
import os
import torch
import torch.nn as nn
import pandas as pd
import joblib  # scalerの保存・読み込みにjoblibを使うのが一般的です
from sklearn.preprocessing import StandardScaler
import beforeinfo_to_features as bf
from roi_util import ROIPredictor
from DualHeadRanker import DualHeadRanker
import traceback
import get_limit
from datetime import datetime, timedelta, timezone

# --- Global inference options ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.80

# LINE SDK
try:
    from linebot import LineBotApi
    from linebot.models import TextSendMessage
except Exception:
    LineBotApi = None
    TextSendMessage = None


# --- グローバル設定 (変更なし) ---
set_global_options(max_instances=10)
initialize_app()

# Firestore (Admin SDK)
DB = firestore.client()

# PyTorchモデルを読み込むヘルパー関数（場ごと）
def load_pytorch_model(jcd: str | None = None):
    class _RankOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, *args, **kwargs):
            _, rank_pred, _ = self.base(*args, **kwargs)
            return rank_pred / TEMPERATURE

    # 検索対象のディレクトリを場別→共通の順で決定
    search_dirs = []
    if jcd:
        jcd_models_dir = os.path.join("models", jcd)
        if os.path.isdir(jcd_models_dir):
            search_dirs.append(jcd_models_dir)
    search_dirs.append("models")  # フォールバック

    model_path = None
    model_list = []
    for d in search_dirs:
        try:
            model_list = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pth")]
        except FileNotFoundError:
            model_list = []
        if model_list:
            model_path = sorted(model_list)[-1]
            break

    if not model_path:
        raise FileNotFoundError("No model checkpoint (.pth) found under per-venue or default 'models' directory.")

    state = torch.load(model_path, map_location="cpu")

    # find the correct weight key for boat_fc
    key_candidates = [k for k in state.keys() if k.endswith("boat_fc.weight")]
    if not key_candidates:
        raise KeyError("'boat_fc.weight' not found in state dict keys: " + ", ".join(state.keys()))
    w = state[key_candidates[0]]
    boat_in_from_ckpt = w.shape[1]  # checkpoint input dim (in_features)

    probe = DualHeadRanker(boat_in=boat_in_from_ckpt)
    if not hasattr(probe, "boat_fc") or not hasattr(probe.boat_fc, "in_features"):
        raise RuntimeError("DualHeadRanker must define boat_fc with in_features")
    probe_in = int(probe.boat_fc.in_features)
    target_in = int(boat_in_from_ckpt)
    extra = probe_in - target_in

    if extra > 0:
        corrected_boat_in = target_in - extra
        if corrected_boat_in <= 0:
            raise RuntimeError(f"Computed corrected_boat_in={corrected_boat_in} is not positive (extra={extra}, target_in={target_in}).")
        model = DualHeadRanker(boat_in=corrected_boat_in)
    else:
        model = DualHeadRanker(boat_in=boat_in_from_ckpt)

    print(f"[load] ({jcd or 'default'}) ckpt boat_fc.in_features={target_in}; class adds extra={extra}; using boat_in={model.boat_fc.in_features - max(extra,0)} → model.boat_fc.in_features={model.boat_fc.in_features}")

    model.load_state_dict(state, strict=True)

    model = model.to(DEVICE)
    rank_model = _RankOnly(model)
    return rank_model

# スケーラー読み込み（場ごと対応）
def load_scaler(jcd: str | None = None):
    search_dirs = []
    if jcd:
        jcd_scalers_dir = os.path.join("scalers", jcd)
        if os.path.isdir(jcd_scalers_dir):
            search_dirs.append(jcd_scalers_dir)
    search_dirs.append("scalers")  # フォールバック

    scaler_path = None
    for d in search_dirs:
        try:
            scaler_list = [f for f in os.listdir(d) if f.endswith(".joblib")]
        except FileNotFoundError:
            scaler_list = []
        if scaler_list:
            latest_scaler_file = sorted(scaler_list)[-1]
            scaler_path = os.path.join(d, latest_scaler_file)
            break

    if not scaler_path:
        raise FileNotFoundError("No scaler file (.joblib) found under per-venue or default 'scalers' directory.")

    print(f"Using scaler: {scaler_path}")
    return joblib.load(scaler_path)

def send_line_message(text: str) -> None:
    if not LINE_ACCESS_TOKEN or not LINE_TO_USER_ID:
        print("[LINE] 環境変数 LINE_CHANNEL_ACCESS_TOKEN / LINE_TO_USER_ID が未設定のため送信をスキップします。")
        return
    if LineBotApi is None:
        print("[LINE] line-bot-sdk がインポートできませんでした（requirements.txt に line-bot-sdk を追加してください）。")
        return

    try:
        api = LineBotApi(LINE_ACCESS_TOKEN)
        api.push_message(LINE_TO_USER_ID, TextSendMessage(text=text))
        print("[LINE] push_message 成功")
    except Exception:
        print("[LINE] push_message 失敗")
        traceback.print_exc()

# --- Per-venue caches ---
MODEL_CACHE: dict[str, nn.Module] = {}
SCALER_CACHE: dict[str, StandardScaler] = {}
PREDICTOR_CACHE: dict[str, ROIPredictor] = {}

NUM_COLS = ["air_temp", "wind_speed", "wave_height", "water_temp", "wind_sin", "wind_cos"]


def get_predictor_for_jcd(jcd: str | None) -> ROIPredictor:
    """場コードごとにモデル/スケーラー/予測器をキャッシュして返す。jcd が None の場合は共通を使う。"""
    key = jcd or "__default__"
    if key in PREDICTOR_CACHE:
        return PREDICTOR_CACHE[key]

    # lazy load
    model = MODEL_CACHE.get(key)
    if model is None:
        model = load_pytorch_model(jcd=jcd)
        MODEL_CACHE[key] = model

    scaler = SCALER_CACHE.get(key)
    if scaler is None:
        scaler = load_scaler(jcd=jcd)
        SCALER_CACHE[key] = scaler

    predictor = ROIPredictor(model=model, scaler=scaler, num_cols=NUM_COLS, device=DEVICE, batch_size=512)
    PREDICTOR_CACHE[key] = predictor
    print(f"[predictor] initialized for jcd={jcd or 'default'} (cache size: {len(PREDICTOR_CACHE)})")
    return predictor

# --- グローバル変数の定義 ---
# コンテナ起動時に一度だけ実行される
try:
    print("--- Starting to initialize globals (lazy per-venue loading) ---")
    # 予測器は場ごとに遅延ロードするので、ここではロードしない
    LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    LINE_TO_USER_ID = os.getenv("LINE_TO_USER_ID")
    if not LINE_ACCESS_TOKEN or not LINE_TO_USER_ID:
        print("[WARNING] LINE_CHANNEL_ACCESS_TOKEN or LINE_TO_USER_ID environment variable is not set. LINE notifications will be disabled.")
    else:
        print("[INFO] LINE_CHANNEL_ACCESS_TOKEN and LINE_TO_USER_ID are set. LINE notifications are enabled.")
    print("--- Global objects initialized (device={}, lazy per-venue caches active). ---".format(DEVICE))
except Exception as e:
    import traceback
    print(f"!!!!!!!! AN ERROR OCCURRED DURING GLOBAL LOADING !!!!!!!")
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR MESSAGE: {e}")
    print("--- TRACEBACK ---")
    traceback.print_exc()

# 通知の実行タイミング（締切何分前か）を環境変数で設定
BEFORE_MINUTES = int(os.getenv("RACE_NOTIFY_BEFORE_MINUTES", "5"))

@https_fn.on_request(memory=MemoryOption.GB_2)
def check_race_notifications(req: https_fn.Request) -> https_fn.Response:
    """
    Cloud Scheduler 等から定期的に呼び出されるハンドラー。
    現在時刻がレース締切の BEFORE_MINUTES 分前〜締切時刻までの範囲に入っていれば
    on_request_example(None) を実行してLINE通知を行う。
    """
    for i in (19, 20):
        jcd = f"0{i}" if i < 10 else str(i)
        # --- 1日1回だけダウンロードするための日次キャッシュ（Firestore） ---
        jst = timezone(timedelta(hours=9))
        now_jst = datetime.now(jst)
        today_hd = now_jst.strftime("%Y%m%d")
        daily_doc_id = f"{jcd}_{today_hd}"

        # まずは Firestore の日次キャッシュを試す
        closing_times = None
        try:
            daily_doc = DB.collection("closing_time_daily").document(daily_doc_id).get()
            if daily_doc.exists:
                payload = daily_doc.to_dict()
                closing_times = payload.get("items") or []
                print(f"[cache] Using cached closing times for {daily_doc_id} (count={len(closing_times)})")
        except Exception as e:
            print(f"[firestore] failed to read daily cache for {daily_doc_id}: {e}")

        # キャッシュが無ければダウンロード → Firestore に保存
        if not closing_times:
            try:
                closing_times = get_limit.extract_closing_times(jcd=jcd)
                DB.collection("closing_time_daily").document(daily_doc_id).set({
                    "jcd": jcd,
                    "hd": today_hd,
                    "count": len(closing_times),
                    "items": closing_times,
                    "fetchedAt": firestore.SERVER_TIMESTAMP,
                }, merge=True)
                # 任意: 履歴用スナップショットも作成（デバッグ・差分検知用）
                snap_id = f"{daily_doc_id}_{now_jst.strftime('%H%M%S')}"
                DB.collection("closing_time_snapshots").document(snap_id).set({
                    "jcd": jcd,
                    "hd": today_hd,
                    "count": len(closing_times),
                    "items": closing_times,
                    "fetchedAt": firestore.SERVER_TIMESTAMP,
                }, merge=True)
                print(f"[fetch] Downloaded and cached closing times for {daily_doc_id} (count={len(closing_times)})")
            except Exception as e:
                msg = f"[check] Failed to fetch closing times: {e}"
                print(msg)
                return https_fn.Response(msg, status=500)
        print('ナウ', now_jst.isoformat())
        print('締切一覧', closing_times)
        for item in closing_times:
            if item == "Not held":
                continue
            iso = item.get("iso_jst")
            rno = item.get("rno")

            # Firestore: upsert closing time document
            doc_id = f"{item.get('jcd')}_{item.get('hd')}_{item.get('rno')}"
            try:
                DB.collection("closing_times").document(doc_id).set({
                    "jcd": item.get("jcd"),
                    "hd": item.get("hd"),
                    "rno": item.get("rno"),
                    "time": item.get("time"),
                    "iso_jst": item.get("iso_jst"),
                    "href": item.get("href"),
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                }, merge=True)
            except Exception as e:
                print(f"[firestore] closing_times upsert failed for {doc_id}: {e}")

            if not iso:
                continue
            dt = datetime.fromisoformat(iso)         # JST(+09:00) のawareなdatetime
            now_tz = now_jst.replace(tzinfo=dt.tzinfo)
            notify_start = dt - timedelta(minutes=BEFORE_MINUTES)

            # Firestore: skip if already notified for this race
            notif_doc_id = doc_id
            try:
                notif_doc = DB.collection("race_notifications").document(notif_doc_id).get()
                if notif_doc.exists and notif_doc.to_dict().get("sent"):
                    print(f"[check] Already notified for {notif_doc_id}; skipping.")
                    continue
            except Exception as e:
                print(f"[firestore] notification doc read failed for {notif_doc_id}: {e}")

            print(notify_start, now_tz, dt, rno)

            if notify_start <= now_tz <= dt:
                req = {
                    "jcd": item.get("jcd"),
                    "hd": item.get("hd"),
                    "rno": rno,
                    "closing_time": iso,
                    "fs_notif_doc_id": notif_doc_id,
                }
                on_request_example(req)              # 予測とLINE送信を実行

    return https_fn.Response("End check", status=200)

# --- HTTPリクエストを処理する関数 ---
@https_fn.on_request(memory=MemoryOption.GB_2)
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    fs_notif_doc_id = req.get("fs_notif_doc_id")
    threshold_dict = {
        "19": 0.22,  # 下関
        "20": 0.21,  # 若松
    }
    features_df = bf.main(rno=req.get("rno"), jcd=req.get("jcd"))
    print(f"[predict] Features shape: {features_df.shape}")
    if features_df.empty:
        print("[predict] No rows fetched for the specified period.")

    exclude = []
    for lane in range(1, 7):
        # 実際に存在するカラム名に合わせてください
        exclude.append(f"lane{lane}_bf_course")
        exclude.append(f"lane{lane}_bf_st_time")
        exclude.append(f"lane{lane}_weight")
        # 他にも学習時に使っていない特徴量があれば追加

    # errors='ignore' をつけて、存在しない列があってもエラーにならないようにする
    features_df.drop(columns=exclude, inplace=True, errors="ignore")

    # ★ 場ごとのPREDICTORを取得して使用する
    predictor = get_predictor_for_jcd(req.get("jcd"))
    pred_scores_df = predictor.predict_scores(features_df, include_meta=True)
    pred_probs_df = predictor.predict_win_probs(scores_df=pred_scores_df, include_meta=True)
    exa_df, tri_df = predictor.predict_exotics_topk(scores_df=pred_scores_df, K=10, tau=1.0, include_meta=True)
    print(f"[predict] Prediction completed. Scores shape: {pred_scores_df.shape}, Win probs shape: {pred_probs_df.shape}")
    print(f"[predict] Example Scores:\n{pred_scores_df.head()}")
    print(f"[predict] Example Win Probs:\n{pred_probs_df.head()}")
    print(f"[predict] Example Exactas:\n{exa_df.head()}")
    print(f"[predict] Example Trifectas:\n{tri_df.head()}")
    # tri_dfのprob列の1行目を取得
    if not tri_df.empty:
        top_tri_prob = tri_df.iloc[0]['prob']
        top_trifecta = tri_df.iloc[0]['trifecta']
        threshold = threshold_dict.get(req.get("jcd"))
        if top_tri_prob > threshold:
            jcd = req.get("jcd")
            hd = req.get("hd")
            rno = req.get("rno")
            if jcd and hd and rno:
                send_line_message(f"{jcd}, {hd}, {rno}, Top Trifecta: {top_trifecta} with Probability: {top_tri_prob:.4f}\nhttps://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={hd}")
                print(f"[predict] Top Trifecta: {top_trifecta} with Probability: {top_tri_prob}")
                try:
                    notif_doc_id = fs_notif_doc_id or f"{jcd}_{hd}_{rno}"
                    DB.collection("race_notifications").document(notif_doc_id).set({
                        "jcd": jcd,
                        "hd": hd,
                        "rno": rno,
                        "closing_time": req.get("closing_time"),
                        "sent": True,
                        "sentAt": firestore.SERVER_TIMESTAMP,
                        "tri": top_trifecta,
                        "prob": float(top_tri_prob),
                        "result_url": f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={hd}",
                    }, merge=True)
                except Exception as e:
                    print(f"[firestore] failed to mark notification sent for {notif_doc_id}: {e}")
        else:
            jcd = req.get("jcd")
            hd = req.get("hd")
            rno = req.get("rno")
            try:
                doc_id = fs_notif_doc_id or f"{jcd}_{hd}_{rno}"
                DB.collection("race_notifications").document(doc_id).set({
                    "jcd": jcd,
                    "hd": hd,
                    "rno": rno,
                    "closing_time": req.get("closing_time"),
                    "sent": False,
                    "lastEvaluatedAt": firestore.SERVER_TIMESTAMP,
                    "topTriProb": float(top_tri_prob),
                }, merge=True)
            except Exception as e:
                print(f"[firestore] failed to update evaluation for {doc_id}: {e}")
            print("[predict] Top trifecta probability is below threshold; no notification sent.")
    else:
        print("[predict] No trifecta predictions available.")

    # TODO: 予測結果をDBに保存したり、レスポンスとして返したりする処理
    print("Prediction completed successfully.")

    return https_fn.Response(f"top_trifecta = {tri_df.iloc[0]['trifecta']}.", status=200)