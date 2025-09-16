import os
import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import requests
from datetime import datetime

def download_off_pred(start_ymd: str, kind: str = "beforeinfo", rno: int = 1) -> str:

    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}?jcd=20"
    current_date = datetime.strptime(start_ymd, "%Y-%m-%d")

    target_ymd = current_date.strftime("%Y%m%d")
    print(target_ymd)
    target_url = f"{base_url}&hd={target_ymd}"
    if kind == "raceindex":
        target_url_no = target_url  # raceindex は rno パラメータ不要
    else:
        target_url_no = f"{target_url}&rno={rno}"
    print(f"▶ 取得中: {target_url_no} ...")
    try:
        response = requests.get(target_url_no, timeout=(5, 30))
        if response.status_code == 200 and "<html" in response.text.lower():            
            return response.text
        else:
            return f"⚠️ 失敗: ステータスコード {response.status_code} または HTML未検出"

    except requests.RequestException as e:
        return f"❌ エラー: {e}"