import os
import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import requests
from datetime import datetime, timedelta
import time

def download_off_pred(start_ymd: str, jcd:str, days: int = 1, interval_sec: int = 1, kind: str = "beforeinfo"):

    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}?jcd={jcd}"
    current_date = datetime.strptime(start_ymd, "%Y%m%d")

    for i in range(days):
        target_ymd = current_date.strftime("%Y%m%d")
        print(target_ymd)
        target_url = f"{base_url}&hd={target_ymd}"
        for j in range(1, 2):
            target_url_no = f"{target_url}&rno={j}"

            # 先に保存先パスを決定し、既に存在する場合はリクエストをスキップ
            file_name = f"{kind}_{jcd}_{target_ymd}_{j}.html"
            dir_name = f"download/{jcd}_off_{kind}_pred_html"
            file_path = os.path.join(dir_name, file_name)
            if os.path.exists(file_path):
                print(f"⏭️ スキップ: 既に存在 {file_name}")
                continue

            print(f"▶ 取得中: {target_url_no} ...")
            try:
                response = requests.get(target_url_no, timeout=(5, 30))
                if response.status_code == 200 and "<html" in response.text.lower():
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"✅ 成功: {file_name} を保存しました。")
                else:
                    print(f"⚠️ 失敗: ステータスコード {response.status_code} または HTML未検出")

            except requests.RequestException as e:
                print(f"❌ エラー: {e}")
            # 次の日へ
        current_date += timedelta(days=1)

if __name__ == "__main__":
    dates =  [
    '20250117', '20250118', '20250119', '20250120', '20250121', '20250122', '20250131', '20250201', '20250202', '20250203',
    '20250204', '20250205', '20250214', '20250215', '20250216', '20250217', '20250218', '20250219', '20250222', '20250223',
    '20250224', '20250225', '20250226', '20250227', '20250303', '20250304', '20250305', '20250306', '20250307', '20250308',
    '20250313', '20250314', '20250315', '20250316', '20250317', '20250321', '20250322', '20250323', '20250324', '20250325',
    '20250326', '20250327', '20250401', '20250402', '20250403', '20250404', '20250405', '20250406', '20250422', '20250423',
    '20250424', '20250425', '20250426', '20250427', '20250501', '20250502', '20250503', '20250504', '20250505', '20250506',
    '20250510', '20250511', '20250512', '20250513', '20250517', '20250518', '20250519', '20250520', '20250521', '20250525',
    '20250526', '20250527', '20250528', '20250529', '20250530', '20250607', '20250608', '20250609', '20250610', '20250611',
    '20250612', '20250617', '20250618', '20250619', '20250620', '20250621', '20250622', '20250626', '20250627', '20250628',
    '20250629', '20250630', '20250704', '20250705', '20250706', '20250707', '20250708', '20250709', '20250715', '20250716',
    '20250717', '20250718', '20250719', '20250720', '20250724', '20250725', '20250726', '20250727', '20250728', '20250729',
    '20250802', '20250803', '20250804', '20250805', '20250806', '20250807', '20250811', '20250812', '20250813', '20250814',
    '20250815', '20250816', '20250824', '20250825', '20250826', '20250827', '20250828', '20250829', '20250906', '20250907',
    '20250908', '20250909', '20250910', '20250911', '20250921', '20250922', '20250923', '20250924', '20250925', '20250926',
    '20251007', '20251008', '20251009', '20251010', '20251011', '20251012', '20251013'
    ]
    # 先頭
    for d in dates:
        download_off_pred(start_ymd=d, jcd="01", days=1, interval_sec=1, kind="beforeinfo")