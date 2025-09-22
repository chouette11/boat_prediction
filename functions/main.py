from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
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

# --- グローバルスコープでモデルとスケーラーを一度だけ読み込む ---

# PyTorchモデルを読み込むヘルパー関数
def load_pytorch_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TEMPERATURE = 0.80

    class _RankOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, *args, **kwargs):
            _, rank_pred, _ = self.base(*args, **kwargs)
            return rank_pred / TEMPERATURE

    # モデルファイルのパスを特定
    model_list = [f for f in os.listdir("models") if f.endswith(".pth")]
    latest = os.path.join("models", sorted(model_list)[-1])
    state = torch.load(latest, map_location="cpu")

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
    extra = probe_in - target_in  # positive if the class adds dims internally

    if extra > 0:
        # compensate the extra so final in_features == target_in
        corrected_boat_in = target_in - extra
        if corrected_boat_in <= 0:
            raise RuntimeError(f"Computed corrected_boat_in={corrected_boat_in} is not positive (extra={extra}, target_in={target_in}).")
        model = DualHeadRanker(boat_in=corrected_boat_in)
    else:
        model = DualHeadRanker(boat_in=boat_in_from_ckpt)

    # sanity log
    print(f"[load] ckpt boat_fc.in_features={target_in}; class adds extra={extra}; using boat_in={model.boat_fc.in_features - max(extra,0)} → model.boat_fc.in_features={model.boat_fc.in_features}")

    model.load_state_dict(state, strict=True)

    model = model.to(device)
    rank_model = _RankOnly(model)
    return rank_model, device

# ★★★ 重要 ★★★
# スケーラーは学習時に作成し、ファイルとして保存しておく必要があります。
# 予測リクエストのたびに .fit() を呼ぶのは間違いです。
# ここでは 'scaler.joblib' という名前で保存されていると仮定します。
def load_scaler():
    scaler_list = [f for f in os.listdir("scalers") if f.endswith(".joblib")]
    if not scaler_list:
        raise FileNotFoundError("No scaler file (.joblib) found in 'scalers' directory.")
    latest_scaler_file = sorted(scaler_list)[-1]
    scaler_path = os.path.join("scalers", latest_scaler_file)
    print(f"Using latest scaler: {scaler_path}")
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

@https_fn.on_request(memory=MemoryOption.GB_2)
def line_send_test(req: https_fn.Request) -> https_fn.Response:
    send_line_message("これはテストメッセージです。")
    return https_fn.Response("Sent test message.", status=200)

# --- グローバル変数の定義 ---
# コンテナ起動時に一度だけ実行される
try:
    print("--- Starting to load global objects... ---")
    RANK_MODEL, DEVICE = load_pytorch_model()
    print("--- PyTorch model loaded successfully. ---")
    SCALER = load_scaler()
    print("--- Scaler loaded successfully. ---")
    NUM_COLS = ["air_temp", "wind_speed", "wave_height", "water_temp", "wind_sin", "wind_cos"]
    PREDICTOR = ROIPredictor(model=RANK_MODEL, scaler=SCALER, num_cols=NUM_COLS, device=DEVICE, batch_size=512)
    print("--- Predictor initialized successfully. ---")
    LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    LINE_TO_USER_ID = os.getenv("LINE_TO_USER_ID")
    if not LINE_ACCESS_TOKEN or not LINE_TO_USER_ID:
        print("[WARNING] LINE_CHANNEL_ACCESS_TOKEN or LINE_TO_USER_ID environment variable is not set. LINE notifications will be disabled.")
    else:
        print("[INFO] LINE_CHANNEL_ACCESS_TOKEN and LINE_TO_USER_ID are set. LINE notifications are enabled.")
    print("--- Global objects loaded without errors! ---")
except Exception as e:
    import traceback
    print(f"!!!!!!!! AN ERROR OCCURRED DURING GLOBAL LOADING !!!!!!!!")
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
    try:
        closing_times = get_limit.extract_closing_times()
    except Exception as e:
        msg = f"[check] Failed to fetch closing times: {e}"
        print(msg)
        return https_fn.Response(msg, status=500)

    # 日本標準時
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    print('ナウ', now_jst.isoformat())
    print('締切一覧', closing_times)
    executed = 0
    for item in closing_times:
        iso = item.get("iso_jst")
        rno = item.get("rno")
        if not iso:
            continue
        dt = datetime.fromisoformat(iso)         # JST(+09:00) のawareなdatetime
        now_tz = now_jst.replace(tzinfo=dt.tzinfo)
        notify_start = dt - timedelta(minutes=BEFORE_MINUTES)
        print(notify_start, now_tz, dt, rno)

        if notify_start <= now_tz <= dt:
            executed += 1
            req = {
                "jcd": item.get("jcd"),
                "hd": item.get("hd"),
                "rno": rno,
                "closing_time": iso,
            }
            on_request_example(req)              # 予測とLINE送信を実行

    if executed:
        return https_fn.Response(f"Executed {executed} notification(s).", status=200)
    return https_fn.Response("No notifications to send at this time.", status=200)

# --- HTTPリクエストを処理する関数 ---
@https_fn.on_request(memory=MemoryOption.GB_2)
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    """
    リクエストごとに実行される処理。
    モデルの読み込みは行わず、予測処理に専念する。
    """
    features_df = bf.main(rno=req.get("rno", 8))  # rnoをリクエストから取得、デフォルトは8
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

    # ★ グローバルに読み込み済みのPREDICTORを再利用する
    pred_scores_df = PREDICTOR.predict_scores(features_df, include_meta=True)
    pred_probs_df = PREDICTOR.predict_win_probs(scores_df=pred_scores_df, include_meta=True)
    exa_df, tri_df = PREDICTOR.predict_exotics_topk(scores_df=pred_scores_df, K=10, tau=1.0, include_meta=True)
    print(f"[predict] Prediction completed. Scores shape: {pred_scores_df.shape}, Win probs shape: {pred_probs_df.shape}")
    print(f"[predict] Example Scores:\n{pred_scores_df.head()}")
    print(f"[predict] Example Win Probs:\n{pred_probs_df.head()}")
    print(f"[predict] Example Exactas:\n{exa_df.head()}")
    print(f"[predict] Example Trifectas:\n{tri_df.head()}")
    # tri_dfのprob列の1行目を取得
    if not tri_df.empty:
        top_tri_prob = tri_df.iloc[0]['prob']
        top_trifecta = tri_df.iloc[0]['trifecta']
        if top_tri_prob > 0.21:  # 確率が1%を超える場合のみ通知
            jcd = req.get("jcd")
            hd = req.get("hd")
            rno = req.get("rno")
            if jcd and hd and rno:
                send_line_message(f"{jcd}, {hd}, {rno}, Top Trifecta: {top_trifecta} with Probability: {top_tri_prob:.4f}\nhttps://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={hd}")
                print(f"[predict] Top Trifecta: {top_trifecta} with Probability: {top_tri_prob}")
        else:
            print("[predict] Top trifecta probability is below threshold; no notification sent.")
    else:
        print("[predict] No trifecta predictions available.")

    # TODO: 予測結果をDBに保存したり、レスポンスとして返したりする処理
    print("Prediction completed successfully.")

    return https_fn.Response(f"top_trifecta = {tri_df.iloc[0]['trifecta']}.", status=200)