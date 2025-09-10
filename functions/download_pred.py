import os
import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import requests
from datetime import datetime, timedelta
import time

def download_off_pred(start_ymd: str, days: int = 1, interval_sec: int = 1, kind: str = "beforeinfo"):
    
    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}?jcd=20"
    current_date = datetime.strptime(start_ymd, "%Y%m%d")

    for i in range(days):
        target_ymd = current_date.strftime("%Y%m%d")
        print(target_ymd)
        target_url = f"{base_url}&hd={target_ymd}"
        for j in range(1, 2):
            target_url_no = f"{target_url}&rno={j}"
            print(f"▶ 取得中: {target_url_no} ...")
            try:
                response = requests.get(target_url_no, timeout=(5, 30))
                if response.status_code == 200 and "<html" in response.text.lower():
                    file_name = f"wakamatsu_{kind}_20_{target_ymd}_{j}.html"
                    dir_name = f"download/wakamatsu_off_{kind}_pred_html"
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    file_path = os.path.join(dir_name, file_name)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    
                    # ファイルの内容をreturn
                    return response.text
                    print(f"✅ 成功: {file_name} を保存しました。")
                else:
                    return f"⚠️ 失敗: ステータスコード {response.status_code} または HTML未検出"

            except requests.RequestException as e:
                return f"❌ エラー: {e}"

            # インターバル
            if i < days - 1:
                print(f"⏳ {interval_sec}秒待機中...\n")
                time.sleep(interval_sec)

            # 次の日へ
            current_date += timedelta(days=1)
    return "すべてのデータを正常に取得しました。"