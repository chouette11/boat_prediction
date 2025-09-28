from pathlib import Path
import download_pred
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone, timedelta
import json, re

def extract_closing_times(jcd: str) -> list[dict]:
    """
    PC版 'レース一覧' ページのHTMLから
    各Rの締切予定時刻・rno・jcd・hd を抽出する。

    Returns: list[dict]
      - rno: レース番号(int)
      - time: 'HH:MM'
      - iso_jst: 'YYYY-MM-DDTHH:MM:SS+09:00'（hdが取れた時のみ）
      - jcd: 場コード（文字列）
      - hd: 開催日(YYYYMMDD)
      - href: 行のレース詳細リンク（相対）
    """
    # 日本標準時
    today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    html = download_pred.download_off_pred(jcd, today, kind="raceindex")
    print(f'html {html[100:300]}...')  # 先頭300文字を表示
    soup = BeautifulSoup(html, "html.parser")

    results = []
    # table > tbody > tr を走査（tbodyは複数に分かれている想定）
    for tr in soup.select("div.table1 table tbody > tr"):
        # 列: [レース, 締切予定時刻, 投票/発売終了, ...]
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 2:
            continue

        # 1列目のリンクから rno/jcd/hd を取得
        a = tds[0].select_one('a[href*="racelist"]')
        if not a:
            continue
        href = a.get("href", "")
        q = parse_qs(urlparse(href).query)
        rno = q.get("rno", [None])[0]
        hd  = q.get("hd",  [None])[0]

        # レース番号をint化（テキスト "1R" からの抽出でもOK）
        if rno is None:
            m = re.search(r"(\d+)\s*R", a.get_text(strip=True))
            rno_i = int(m.group(1)) if m else None
        else:
            rno_i = int(rno)

        # 2列目テキストが締切予定時刻（例: '18:36'）
        time_str = tds[1].get_text(strip=True)

        # hd と時刻から ISO（JST）を組み立て（取れない場合は None）
        iso_jst = None
        if hd and re.fullmatch(r"\d{8}", hd) and re.fullmatch(r"\d{1,2}:\d{2}", time_str):
            dt = datetime.strptime(f"{hd} {time_str}", "%Y%m%d %H:%M")
            iso_jst = dt.strftime("%Y-%m-%dT%H:%M:%S+09:00")

        results.append({
            "rno": rno_i,
            "time": time_str,
            "iso_jst": iso_jst,
            "jcd": jcd,
            "hd": hd,
            "href": href,
        })

    # レース番号順に整列
    results.sort(key=lambda x: (x["rno"] if x["rno"] is not None else 0))
    return results


if __name__ == "__main__":
    path = "/mnt/data/レース一覧｜BOAT RACE オフィシャルウェブサイト.html"
    data = extract_closing_times(path)
    print(json.dumps(data, ensure_ascii=False, indent=2))