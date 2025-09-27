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
        for j in range(1, 13):
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
                    print(f"✅ 成功: {file_name} を保存しました。")
                else:
                    print(f"⚠️ 失敗: ステータスコード {response.status_code} または HTML未検出")

            except requests.RequestException as e:
                print(f"❌ エラー: {e}")
            # 次の日へ
        current_date += timedelta(days=1)

if __name__ == "__main__":
    dates = [
    "20250113",
    "20250114",
    "20250115",
    "20250116",
    "20250117",
    "20250118",
    "20250119",
    "20250126",
    "20250127",
    "20250128",
    "20250129",
    "20250130",
    "20250201",
    "20250202",
    "20250203",
    "20250204",
    "20250206",
    "20250209",
    "20250210",
    "20250211",
    "20250212",
    "20250216",
    "20250217",
    "20250218",
    "20250219",
    "20250220",
    "20250226",
    "20250227",
    "20250228",
    "20250301",
    "20250302",
    "20250303",
    "20250312",
    "20250313",
    "20250314",
    "20250315",
    "20250317",
    "20250319",
    "20250320",
    "20250321",
    "20250325",
    "20250326",
    "20250327",
    "20250328",
    "20250329",
    "20250330",
    "20250418",
    "20250419",
    "20250420",
    "20250421",
    "20250424",
    "20250425",
    "20250426",
    "20250427",
    "20250428",
    "20250429",
    "20250504",
    "20250505",
    "20250506",
    "20250507",
    "20250508",
    "20250512",
    "20250513",
    "20250514",
    "20250515",
    "20250517",
    "20250518",
    "20250519",
    "20250520",
    "20250523",
    "20250524",
    "20250525",
    "20250526",
    "20250603",
    "20250604",
    "20250605",
    "20250606",
    "20250607",
    "20250608",
    "20250613",
    "20250614",
    "20250615",
    "20250616",
    "20250617",
    "20250619",
    "20250620",
    "20250621",
    "20250622",
    "20250623",
    "20250624",
    "20250628",
    "20250629",
    "20250630",
    "20250701",
    "20250702",
    "20250703",
    "20250707",
    "20250708",
    "20250709",
    "20250710",
    "20250711",
    "20250714",
    "20250715",
    "20250716",
    "20250717",
    "20250722",
    "20250723",
    "20250724",
    "20250725",
    "20250728",
    "20250801",
    "20250802",
    "20250803",
    "20250804",
    "20250806",
    "20250807",
    "20250808",
    "20250809",
    "20250811",
    "20250812",
    "20250815",
    "20250816",
    "20250817",
    "20250818",
    "20250819",
    "20250820",
    "20250826",
    "20250827",
    "20250828",
    "20250829",
    "20250830",
    "20250831",
    "20250904",
    "20250905",
    "20250906",
    "20250907",
    "20250908",
    "20250909",
    "20250914",
    "20250915",
    "20250916",
    "20250917",
    "20250918",
    "20250919",
    "20250922",
    ]
    # 先頭
    for d in dates[130:140]:
        download_off_pred(start_ymd=d, days=1, interval_sec=1, kind="beforeinfo")