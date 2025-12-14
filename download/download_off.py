import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

jcd_to_en = {
    "01": "Kiryu",
    "04": "Heiwajima",
    "07": "Gamagori",
    "12": "Suminoe",
    "15": "Marugame",
    "19": "Shimonoseki",
    "20": "Wakamatsu",
    "24": "Omura"
}

def _make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    return s

def _list_existing_rnos(session, ymd, jcd):
    """
    raceindex を見て、その日に実在する R のみを返す（1..12 の中から抽出）
    """
    url = f"https://boatrace.jp/owpc/pc/race/raceindex?hd={ymd}&jcd={jcd}"
    try:
        r = session.get(url, timeout=(5, 15))
        if r.status_code != 200:
            return []
        # 軽量に rno=1..12 の出現を拾う
        html = r.text
        rnos = set(int(m) for m in re.findall(r"rno=(\d{1,2})", html))
        return sorted([r for r in rnos if 1 <= r <= 12])
    except requests.RequestException:
        return []

def _fetch_and_save(session, kind, ymd, jcd, en, rno, out_root="download", overwrite=False):
    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}"
    url = f"{base_url}?jcd={jcd}&hd={ymd}&rno={rno}"
    dir_name = os.path.join(out_root, f"{en.lower()}_off_{kind}_html")
    os.makedirs(dir_name, exist_ok=True)
    file_name = f"{en.lower()}_{kind}_{jcd}_{ymd}_{rno}.html"
    path = os.path.join(dir_name, file_name)

    if (not overwrite) and os.path.exists(path):
        return ("skip", path)

    try:
        resp = session.get(url, timeout=(5, 30))
        if resp.status_code == 200 and "<html" in resp.text.lower():
            with open(path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            return ("ok", path)
        return ("bad", f"status={resp.status_code}")
    except Exception as e:
        return ("err", str(e))

def download_off(start_ymd: str, days: int = 3, interval_sec: int = 1, kind: str = "result",
                 max_workers: int = 8, out_root: str = "download", overwrite: bool = False, verbose: bool = False):
    
    base_url = f"https://boatrace.jp/owpc/pc/race/{kind}"
    current_date = datetime.strptime(start_ymd, "%Y%m%d")
    session = _make_session()

    for i in range(days):
        ymd = current_date.strftime("%Y%m%d")
        day_tasks = []

        # 開催のある場とRだけを列挙
        for jcd, en in jcd_to_en.items():
            rnos = _list_existing_rnos(session, ymd, jcd)
            if not rnos:
                if verbose:
                    print(f"[{ymd}] jcd={jcd} ({en}) 開催なし/取得対象なし")
                continue
            for rno in rnos:
                day_tasks.append((jcd, en, rno))

        if verbose:
            print(f"\n==== {ymd} ==== 取得タスク数: {len(day_tasks)}")

        if not day_tasks:
            # 次の日へ
            current_date += timedelta(days=1)
            if i < days - 1 and interval_sec > 0:
                time.sleep(interval_sec)
            continue

        # I/O バウンド → スレッドで並列化
        saved = skipped = errors = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_fetch_and_save, session, kind, ymd, jcd, en, rno, out_root, overwrite)
                    for (jcd, en, rno) in day_tasks]
            for fut in as_completed(futs):
                status, info = fut.result()
                if status == "ok":
                    saved += 1
                    if verbose:
                        print(f"✅ {info}")
                elif status == "skip":
                    skipped += 1
                    if verbose:
                        print(f"⏭  {info}")
                else:
                    errors += 1
                    if verbose:
                        print(f"⚠️ {status}: {info}")

        print(f"[{ymd}] done. saved={saved}, skip={skipped}, err={errors}")

        # 次の日へ
        current_date += timedelta(days=1)
        if i < days - 1 and interval_sec > 0:
            time.sleep(interval_sec)

if __name__ == "__main__":
    download_off("20251101", days=30, interval_sec=0, kind="beforeinfo", max_workers=8, verbose=False)