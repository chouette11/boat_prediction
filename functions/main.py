# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app

from google.cloud import scheduler_v1
from google.protobuf import field_mask_pb2

import datetime
import download_pred
import wakamatsu_off_beforeinfo_html_to_csv
from roi_util import ROIPredictor

# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-function
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator, e.g. @https_fn.on_request(max_instances=5).
set_global_options(max_instances=10)

initialize_app()

@https_fn.on_request()
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    today = datetime.datetime.now()
    html = download_pred.download_off_pred(today.strftime("%Y%m%d"))
    racers_df, weather_df = wakamatsu_off_beforeinfo_html_to_csv.parse_boat_race_html(html, is_pred=True)

    df_recent.drop(columns=exclude, inplace=True, errors="ignore")

    if df_recent.empty:
        print("[predict] No rows fetched for the specified period.")

    print(f"[predict] Loaded {len(df_recent)} rows ({start_date} – {today}).")
    print(f"columns: {', '.join(df_recent.columns)}")

    # ------------------------------
    # ROIPredictor でスコア＆確率を一括生成
    # ------------------------------
    predictor = ROIPredictor(model=rank_model, scaler=scaler,
                            num_cols=NUM_COLS, device=device, batch_size=512)

    # (1) スコア（logits）: lane1_score..lane6_score (+ メタ列) を保存
    pred_scores_df = predictor.predict_scores(df_recent,
                                            include_meta=True,
                                            save_to="artifacts/pred_scores.csv")

    # (2) 勝率＆フェアオッズを保存
    pred_probs_df = predictor.predict_win_probs(scores_df=pred_scores_df,
                                                include_meta=True,
                                                save_to="artifacts/pred_win_probs.csv")
    # (3) 馬単/三連単の TOP‑K（PL 方式）を保存
    exa_df, tri_df = predictor.predict_exotics_topk(scores_df=pred_scores_df,
                                                    K=10,
                                                    tau=5.0,
                                                    include_meta=True,
                                                    save_exacta="artifacts/pred_exacta_topk.csv",
                                                    save_trifecta="artifacts/pred_trifecta_topk.csv")
    
    return https_fn.Response(mes)




# Google Cloudプロジェクトの情報を設定
PROJECT_ID = "your-gcp-project-id"  # ご自身のプロジェクトIDに書き換えてください
LOCATION_ID = "asia-northeast1"     # ご自身の関数のリージョンに書き換えてください

@https_fn.on_request()
def update_other_function_schedule(req: https_fn.Request) -> https_fn.Response:
    """
    HTTPリクエストを受け取り、別のスケジュール関数の時刻設定を更新する。
    リクエスト例: {"target_function": "myScheduledFunction", "new_schedule": "0 10 * * *"}
    """
    
    # リクエストからパラメータを取得
    try:
        data = req.get_json()
        target_function_name = data["target_function"]
        new_cron_schedule = data["new_schedule"] # "分 時 日 月 曜日" のcron形式
    except Exception as e:
        return https_fn.Response(f"リクエストの形式が正しくありません: {e}", status=400)

    # Cloud Schedulerのクライアントを初期化
    client = scheduler_v1.CloudSchedulerClient()

    # Firebase Functionsが作成するジョブ名は特定の形式になる
    # firebase-schedule-[関数名]-[リージョン]
    job_id = f"firebase-schedule-{target_function_name}-{LOCATION_ID}"
    
    try:
        # 更新するジョブのフルパスを取得
        job_name = client.job_path(PROJECT_ID, LOCATION_ID, job_id)
        
        # 更新内容（新しいスケジュール）を準備
        job = {
            "name": job_name,
            "schedule": new_cron_schedule
        }

        # どのフィールドを更新するかを指定（今回はスケジュールのみ）
        update_mask = field_mask_pb2.FieldMask(paths=["schedule"])
        
        # ジョブ更新APIを呼び出し
        request = scheduler_v1.UpdateJobRequest(
            job=job,
            update_mask=update_mask,
        )
        client.update_job(request=request)
        
        message = f"ジョブ '{job_id}' のスケジュールを '{new_cron_schedule}' に更新しました。"
        print(message)
        return https_fn.Response(message)

    except Exception as e:
        message = f"ジョブの更新に失敗しました: {e}"
        print(message)
        return https_fn.Response(message, status=500)