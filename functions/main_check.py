# from firebase_functions import https_fn
# from firebase_functions.options import set_global_options
# from firebase_admin import initialize_app, firestore
# from firebase_functions.options import MemoryOption
import os
import torch
import torch.nn as nn
import pandas as pd
import joblib  # scalerの保存・読み込みにjoblibを使うのが一般的です
from sklearn.preprocessing import StandardScaler
import beforeinfo_to_features_check as bf
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
# set_global_options(max_instances=10)
# initialize_app()

# # Firestore (Admin SDK)
# DB = firestore.client()

# PyTorchモデルを読み込むヘルパー関数（場ごと）
def load_pytorch_model(jcd):
    class _RankOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, *args, **kwargs):
            _, rank_pred, _ = self.base(*args, **kwargs)
            return rank_pred / TEMPERATURE

    # 検索対象のディレクトリを場別→共通の順で決定
    search_dirs = []
    yyyymm = datetime.now().strftime("%Y%m")
    if jcd:
        jcd_models_dir = os.path.join(f"models_{yyyymm}", jcd)
        if os.path.isdir(jcd_models_dir):
            search_dirs.append(jcd_models_dir)
    search_dirs.append(f"models_{yyyymm}")  # フォールバック

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
def load_scaler(jcd):
    search_dirs = []
    yyyymm = datetime.now().strftime("%Y%m")
    if jcd:
        jcd_scalers_dir = os.path.join(f"scalers_{yyyymm}", jcd)
        if os.path.isdir(jcd_scalers_dir):
            search_dirs.append(jcd_scalers_dir)
    search_dirs.append(f"scalers_{yyyymm}")  # フォールバック

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
SANRENTAN_RESULT_CACHE: dict[str, pd.DataFrame] = {}

NUM_COLS = ["air_temp", "wind_speed", "wave_height", "water_temp", "wind_sin", "wind_cos"]

# Columns to exclude from features for all races
EXCLUDE_COLS = [
    f"lane{lane}_{suffix}"
    for lane in range(1, 7)
    for suffix in ("bf_course", "bf_st_time", "weight")
]


def get_predictor_for_jcd(jcd) -> ROIPredictor:
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

def is_hit_trio(pred_trifecta: str, true_trifecta: str) -> bool:
    """3連単の予測と実際の結果を比較して、3連複が当たっているかどうかを判定する関数"""
    pred_set = set(pred_trifecta.split('-'))
    true_set = set(true_trifecta.split('-'))
    return pred_set == true_set



if __name__ == "__main__":
    import functions.sanrentan_util as su
    threshold_dict = {
        "01": 0.20,  # 桐生
        "07": 0.16,  # 蒲郡
        "12": 0.116,  # 住之江
        "15": 0.35,  # 丸亀
        "19": 0.27,  # 下関
        "20": 0.22,  # 若松
        "24": 0.15,  # 大村
    }
    jcd_name_dict = {
        "01": "桐 生",
        "07": "蒲 郡",
        "12": "住之江",
        "15": "丸 亀",
        "19": "下 関",
        "20": "若 松",
        "24": "大 村",
    }
    dir_path = '../download/01_off_beforeinfo_pred_html'
    # 行ごとの結果を一時的に保持するリスト（最後に一括でconcatしてO(n^2)を回避）
    result_rows: list[pd.DataFrame] = []
    for beforeinfo_filename in os.listdir(dir_path):
        html_path = os.path.join(dir_path, beforeinfo_filename)
        basename = os.path.basename(beforeinfo_filename).split('.')[0]
        rno = int(basename.split('_')[-1])  # ファイル名からrnoを抽出
        rno = f'{rno}' if rno >= 10 else f'0{rno}'
        jcd = basename.split('_')[-3]         # ファイル名からjcdを
        hd = basename.split('_')[-2]         # ファイル名からhdを
        features_df = bf.main(rno=rno, jcd=jcd, path=html_path)
        if features_df is None or features_df.empty:
            print(f"[predict] No features generated for jcd={jcd}, hd={hd}, rno={rno}. Skipping.")
            continue
        # print(f"[predict] Features shape: {features_df.shape}")

        # errors='ignore' をつけて、存在しない列があってもエラーにならないようにする
        features_df.drop(columns=EXCLUDE_COLS, inplace=True, errors="ignore")

        try:
            # ★ 場ごとのPREDICTORを取得して使用する
            predictor = get_predictor_for_jcd(jcd)
            pred_scores_df = predictor.predict_scores(features_df, include_meta=True)
            pred_probs_df = predictor.predict_win_probs(scores_df=pred_scores_df, include_meta=True)
            exa_df, tri_df = predictor.predict_exotics_topk(scores_df=pred_scores_df, K=10, tau=1.0, include_meta=True)
            # tri_dfのprob列の1行目を取得
            if not tri_df.empty:
                top_tri_prob = tri_df.iloc[0]['prob']
                top_trifecta = tri_df.iloc[0]['trifecta']
                threshold = threshold_dict.get(jcd)
                csv_path = f"../model/artifacts/{jcd_name_dict.get(jcd)}/eval_features_recent_{jcd_name_dict.get(jcd)}.csv"
                # hdをyyyy-mm-dd形式に変換
                hd_formatted = f"{hd[:4]}-{hd[4:6]}-{hd[6:]}"
                race_key = f"{hd_formatted}-{rno}-{jcd}"
                print(f"[predict] Race Key: {race_key}")

                cache_key = jcd
                if cache_key not in SANRENTAN_RESULT_CACHE:
                    # 一度だけCSVを読み、sanrentanテーブルを作成してキャッシュ
                    src_df = su._ensure_dataframe(csv_path)
                    sanrentan_df = su.compute_sanrentan(src_df)[["race_key", "sanrentan"]].drop_duplicates("race_key")
                    SANRENTAN_RESULT_CACHE[cache_key] = sanrentan_df

                sanrentan_df = SANRENTAN_RESULT_CACHE[cache_key]
                df_true = sanrentan_df.loc[sanrentan_df["race_key"] == race_key].reset_index(drop=True)
                # print(f"[predict] Top Trifecta: {top_trifecta} with Probability: {top_tri_prob}")
                # print(df_true)

                # df_true を元に1レース分の行データを構築し、リストに貯めておく
                row = df_true.copy()
                row['pred'] = top_trifecta
                row['pred_prob'] = top_tri_prob
                row['trifecta_is_hit'] = (df_true['sanrentan'].values[0] == top_trifecta)
                row['trio_is_hit'] = is_hit_trio(top_trifecta, df_true['sanrentan'].values[0])

                # is_threshold は従来通り、条件を満たしたときだけ True を立て、それ以外はNaNのまま
                if top_tri_prob > threshold:
                    # print(f"[predict] {jcd}, {hd}, {rno}, Top Trifecta: {top_trifecta} with Probability: {top_tri_prob:.4f}\nhttps://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={hd}")
                    row['is_threshold'] = True

                result_rows.append(row)
            else:
                print("[predict] No trifecta predictions available.")
        except Exception as e:
            print(f"[predict][ERROR] Skipping file {beforeinfo_filename} due to error: {e}")
            import traceback as _tb
            _tb.print_exc()
            continue

    # 予測結果を一括でDataFrameに変換し、元の列順を維持してCSVに保存
    output_columns = ["race_key", "sanrentan", "pred", "pred_prob", "is_threshold",
                      "trifecta_is_hit", "trio_is_hit", "trifecta_odds", "trio_odds"]
    if result_rows:
        df = pd.concat(result_rows, ignore_index=True)
        # 不足カラムはNaNで補完しつつ、元の列順に揃える
        df = df.reindex(columns=output_columns)
    else:
        df = pd.DataFrame(columns=output_columns)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'{jcd}_predictions_{now}.csv', index=False)


