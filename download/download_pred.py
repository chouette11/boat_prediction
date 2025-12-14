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
        for j in range(1, 13):
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
    dates = [
    '20250702','20250703','20250704','20250705','20250706','20250707',
    '20250711','20250712','20250713','20250714','20250715','20250716',
    '20250721','20250722','20250723','20250724','20250725','20250726',
    '20250729','20250731',
    '20250801','20250802','20250803',
    '20250807','20250808','20250809','20250810','20250811','20250812',
    '20250816','20250817','20250818','20250819','20250820','20250821',
    '20250825','20250826','20250827','20250828','20250829','20250830',
    '20250904','20250906','20250907','20250908','20250909','20250910',
    '20250913','20250914','20250915','20250916','20250917','20250918',
    '20250922','20250923','20250924','20250925','20250926','20250927',
    '20251001','20251002','20251003','20251004','20251005',
]
    
    # 先頭
    for d in dates:
        download_off_pred(start_ymd=d, jcd="04", days=1, interval_sec=1, kind="beforeinfo")