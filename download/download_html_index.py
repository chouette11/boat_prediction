import requests
from datetime import datetime, timedelta
import time

def fetch_results(start_ym: str, months: int = 3, interval_sec: int = 3):
    """
    指定した年月から開始して、指定月数分のHTMLを取得（インターバル付き）

    :param start_ym: 開始年月（例：'202407'）
    :param months: 取得したい月数（例：3なら3か月分）
    :param interval_sec: 各リクエストの待機秒数（例：3秒）
    """
    base_url = "https://kyotei.sakura.ne.jp/racelist-wakamatsu-20250617.html#7R"
    # current_date = datetime.strptime(start_ym, "%Y%m")

    for i in range(1, 24):
        target_url = f"{base_url}dtl={i}&select={i}.html"
        print(f"▶ 取得中: {target_url} ...")

        try:
            response = requests.get(target_url, timeout=10)
            if response.status_code == 200 and "<html" in response.text.lower():
                file_name = f"wakamatsu_result_{i}.html"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"✅ 成功: {file_name} を保存しました。")
            else:
                print(f"⚠️ 失敗: ステータスコード {response.status_code} または HTML未検出")

        except requests.RequestException as e:
            print(f"❌ エラー: {e}")

        # インターバル
        print(f"⏳ {interval_sec}秒待機中...\n")
        time.sleep(interval_sec)

        # 次の月へ
        # current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)

# 使用例：2024年7月から3ヶ月分取得（3秒間隔）
fetch_results("202401", months=6, interval_sec=3)
