import requests
from datetime import datetime, timedelta
import time
import os

def fetch_results(start_ymd: str, days: int = 3, interval_sec: int = 1):
    kind = "result"
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
        for j in range(1, 12):
            target_url_no = f"{target_url}&rno={j}"
            print(f"▶ 取得中: {target_url_no} ...")
            try:
                response = requests.get(target_url_no, timeout=(5, 30))
                if response.status_code == 200 and "<html" in response.text.lower():
                    file_name = f"wakamatsu_{kind}_{target_ymd}_{j}.html"
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


# for i in range(1, 6):
#     n = 20240106 + i
#     fetch_results(str(n), days=35, interval_sec=0)

def fetch_results2(target_ymd: str, kind: str = "result", no: int = 1):
    """
    指定した年月から開始して、指定月数分のHTMLを取得（インターバル付き）

    :param start_ym: 開始年月（例：'202407'）
    :param months: 取得したい月数（例：3なら3か月分）
    :param interval_sec: 各リクエストの待機秒数（例：3秒）
    """
    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}?jcd=20"

    target_url = f"{base_url}&hd={target_ymd}"
    target_url_no = f"{target_url}&rno={no}"
    print(f"▶ 取得中: {target_url_no} ...")
    try:
        response = requests.get(target_url_no, timeout=(5, 30))
        if response.status_code == 200 and "<html" in response.text.lower():
            file_name = f"wakamatsu_{kind}_{target_ymd}_{no}.html"
            file_path = os.path.join(f"download/wakamatsu_off_{kind}_html", file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"✅ 成功: {file_name} を保存しました。")
        else:
            print(f"⚠️ 失敗: ステータスコード {response.status_code} または HTML未検出")

    except requests.RequestException as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    dirs = [
        "download/wakamatsu_off_raceresult_csv",
        # "download/wakamatsu_off_odds3t_csv",
    ]
    for d in dirs:
        files = os.listdir(d)
        for f in files:
            if 'error' in f:
                print(f"{f}")
                elements = f.split('_')
                kind, ymd, no = elements[1], elements[3], elements[4]
                fetch_results2(f"{ymd}", kind=kind, no=no)

                # # 消す
                # file_path = os.path.join(d, f)
                # print(f"削除: {file_path}")
                # os.remove(file_path)
                
