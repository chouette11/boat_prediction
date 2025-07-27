import requests
from datetime import datetime, timedelta
import time
import os

def fetch_results(start_ymd: str, kind: str, days: int = 3, interval_sec: int = 1):
    """
    指定した年月から開始して、指定月数分のHTMLを取得（インターバル付き）

    :param start_ym: 開始年月（例：'202407'）
    :param months: 取得したい月数（例：3なら3か月分）
    :param interval_sec: 各リクエストの待機秒数（例：3秒）
    """
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
                response = requests.get(target_url_no, timeout=(10, 60))
                if response.status_code == 200 and "<html" in response.text.lower():
                    file_name = f"wakamatsu_{kind}_20_{target_ymd}_{j}.html"
                    # 保存先ディレクトリを作成
                    os.makedirs(f"download/wakamatsu_off_{kind}_html", exist_ok=True)
                    file_path = os.path.join(f"download/wakamatsu_off_{kind}_html", file_name)
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


if __name__ == "__main__":
    kinds = [ "odds3t"]
    start_ymd = "20220101"  # 開始年月日
    days = 700  # 取得したい日数
    interval_sec = 1  # 各リクエストの待機秒数
    for kind in kinds:
        fetch_results(start_ymd, kind, days, interval_sec)
    
                
