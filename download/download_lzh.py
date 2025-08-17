import os, time, hashlib, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, lhafile  # pip install lhafile
import pathlib

BASES = {
    "results": "https://www1.mbrace.or.jp/od2/K/{ym}/k{yymmdd}.lzh",
    "programs":"https://www1.mbrace.or.jp/od2/B/{ym}/b{yymmdd}.lzh",
}
OUT_DIR = "download/lzh"

def url_for(d: dt.date, kind: str):
    ym = d.strftime("%Y%m")
    yymmdd = d.strftime("%y%m%d")
    return BASES[kind].format(ym=ym, yymmdd=yymmdd)

def download_one(url: str, out_path: str, timeout=20):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skip"
    h = requests.head(url, timeout=timeout)
    if h.status_code != 200:
        return f"miss:{h.status_code}"
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1<<15):
            if chunk: f.write(chunk)
    os.replace(tmp, out_path)
    time.sleep(0.25)  # polite pause
    return "ok"


def bulk_fetch(start="2022-01-01", end=None, kinds=("results","programs"), workers=8):
    if end is None:
        end = dt.date(2023, 1, 1).isoformat()
    s = dt.date.fromisoformat(start); e = dt.date.fromisoformat(end)
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        cur = s
        while cur <= e:
            for k in kinds:
                url = url_for(cur, k)
                out_path = os.path.join(OUT_DIR, k, cur.strftime("%Y/%m"), os.path.basename(url))
                futures.append(ex.submit(download_one, url, out_path))
            cur += dt.timedelta(days=1)
        for fu in as_completed(futures):
            _ = fu.result()

# 実行例：全期間（2002-01-01〜今日）
bulk_fetch()

