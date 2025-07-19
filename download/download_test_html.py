import requests
from datetime import datetime, timedelta
import time
import os

def fetch_held_dates(start_ymd: str, kind: str, days: int = 3, interval_sec: int = 1, use_fixed_dates: bool = True) -> list:
    if use_fixed_dates:
        held_dates = [
            # ギラヴァンツ北九州杯（4/21〜4/26）
            "20250421", "20250422", "20250423", "20250424", "20250425", "20250426",
            # スポーツニッポン杯GW特選競走（4/29〜5/4）
            "20250429", "20250430", "20250501", "20250502", "20250503", "20250504",
            # 山口シネマカップ（5/15〜5/20）
            "20250515", "20250516", "20250517", "20250518", "20250519", "20250520",
            # ビッグベアーズカップ（5/20〜5/25）
            "20250520", "20250521", "20250522", "20250523", "20250524", "20250525",
            # パイナップルナイターカップ（5/26〜5/31）
            "20250526", "20250527", "20250528", "20250529", "20250530", "20250531",
            # サッポロビールカップ（6/8〜6/13）
            "20250608", "20250609", "20250610", "20250611", "20250612", "20250613",
            # 唐十杯（6/17〜6/22）
            "20250617", "20250618", "20250619", "20250620", "20250621", "20250622",
            # G3 レディースひめちゃん杯（6/24〜6/29）
            "20250624", "20250625", "20250626", "20250627", "20250628", "20250629",
            # ルーキーS（6/28〜7/3）
            "20250628", "20250629", "20250630", "20250701", "20250702", "20250703",
            # 若松夜王（7/7〜7/11）
            "20250707", "20250708", "20250709", "20250710", "20250711",
            # にっぽん未来PJ競走（7/14〜7/17）
            "20250714", "20250715", "20250716", "20250717"
        ]
        held_dates = sorted(list(dict.fromkeys(held_dates)))
    else:
        base_url = f"https://boatrace.jp/owpc/pc/race/{kind}?jcd=20"
        current_date = datetime.strptime(start_ymd, "%Y%m%d")
        held_dates = []

        for i in range(days):
            target_ymd = current_date.strftime("%Y%m%d")
            target_url = f"{base_url}&hd={target_ymd}&rno=1"
            print(f"▶ チェック中: {target_url} ...")

            try:
                response = requests.get(target_url, timeout=(5, 30))
                if response.status_code == 200 and "レース結果" in response.text:
                    held_dates.append(target_ymd)
                    print(f"✅ 開催日: {target_ymd}")
                else:
                    print(f"❌ 非開催日: {target_ymd}")
            except requests.RequestException as e:
                print(f"⚠️ エラー: {e}")

            time.sleep(interval_sec)
            current_date += timedelta(days=1)

    print("\n📅 開催日一覧:")
    for date in held_dates:
        print(date)
    return held_dates

def download_race_html(held_dates: list, kind: str, save_dir: str, interval_sec: int = 1):
    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}?jcd=20"
    os.makedirs(save_dir, exist_ok=True)
    for target_ymd in held_dates:
        for rno in range(1, 13):
            target_url = f"{base_url}&hd={target_ymd}&rno={rno}"
            print(f"▶ ダウンロード中: {target_url} ...")
            try:
                response = requests.get(target_url, timeout=(5, 30))
                if response.status_code == 200 and "<html" in response.text.lower():
                    file_name = f"wakamatsu_{kind}_20_{target_ymd}_{rno}.html"
                    file_path = os.path.join(save_dir, file_name)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"✅ 成功: {file_name}")
                else:
                    print(f"⚠️ スキップ: status={response.status_code}")
            except requests.RequestException as e:
                print(f"❌ エラー: {e}")
            time.sleep(interval_sec)

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)

    kinds = ["beforeinfo"]
    held_dates = fetch_held_dates("20250401", "result", 100, 0, use_fixed_dates=True)
    for kind in kinds:
        download_race_html(held_dates, kind, f"download/wakamatsu_off_{kind}_html", interval_sec=0)