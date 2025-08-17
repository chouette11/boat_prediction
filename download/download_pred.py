import os
import datetime
from wakamatsu_off_beforeinfo_html_to_csv import parse_boat_race_html as parse_beforeinfo_html
from wakamatsu_off_racelist_html_to_csv import main as parse_racelist_html
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sql.etl import predict_main
import requests
from datetime import datetime, timedelta
import time

def download_off(start_ymd: str, days: int = 3, interval_sec: int = 1, kind: str = "result", is_pred: bool = False):
    
    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}?jcd=20"
    current_date = datetime.strptime(start_ymd, "%Y%m%d")

    for i in range(days):
        target_ymd = current_date.strftime("%Y%m%d")
        print(target_ymd)
        target_url = f"{base_url}&hd={target_ymd}"
        for j in range(7, 8):
            target_url_no = f"{target_url}&rno={j}"
            print(f"▶ 取得中: {target_url_no} ...")
            try:
                response = requests.get(target_url_no, timeout=(5, 30))
                if response.status_code == 200 and "<html" in response.text.lower():
                    file_name = f"wakamatsu_{kind}_20_{target_ymd}_{j}.html"
                    dir_name = f"download/wakamatsu_off_{kind}_html"
                    if is_pred:
                        dir_name = f"download/wakamatsu_off_{kind}_pred_html"
                    file_path = os.path.join(dir_name, file_name)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"✅ 成功: {file_name} を保存しました。")
                else:
                    print(f"⚠️ 失敗: ステータスコード {response.status_code} または HTML未検出")

            except requests.RequestException as e:
                print(f"❌ エラー: {e}")

            # インターバル
            if i < days - 1:
                print(f"⏳ {interval_sec}秒待機中...\n")
                time.sleep(interval_sec)

            # 次の日へ
            current_date += timedelta(days=1)

today = datetime.now()
kinds = ['racelist', 'beforeinfo']
for kind in kinds:
    print(f"Downloading {kind} for today: {today.strftime('%Y%m%d')}")
    download_off(today.strftime("%Y%m%d"), days=1, interval_sec=1, kind=kind, is_pred=True)

for beforeinfo_filename in os.listdir('download/wakamatsu_off_beforeinfo_pred_html'):
    beforeinfo_path = os.path.join('download/wakamatsu_off_beforeinfo_pred_html', beforeinfo_filename)
    parse_beforeinfo_html(beforeinfo_path, is_pred=True)

parse_racelist_html(is_pred=True)

predict_main()