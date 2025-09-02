#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
艇国DB レーサーページ（例: https://boatrace-db.net/racer/...）の HTML を読み込み、
含まれるすべての <table> を出来るだけ網羅的に抽出して CSV に書き出します。

◆ 必要ライブラリ
    pip install beautifulsoup4 lxml pandas

◆ 使い方
    python extract_boatrace_tables.py path/to/racer.html output_dir
"""
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from bs4 import BeautifulSoup


def _clean(cell: str) -> str:
    """セル内文字列の前後空白・改行を除去し、全角スペースも半角に統一"""
    return cell.replace("\u3000", " ").strip()


def _numericize(series: pd.Series) -> pd.Series:
    """
    「%」「.」やカンマ区切り、空文字を含む列を数値に変換。
    変換できない場合は元の文字列のままにしておく。
    """
    s = series.str.replace("%", "", regex=False)           # 1着率 10.0 % → 10.0
    s = s.str.replace(",", "", regex=False)                # 1,259 → 1259
    return pd.to_numeric(s, errors="ignore")


def _racer_profile(soup: BeautifulSoup) -> Dict[str, str]:
    """
    ページの <h1 class="side_title -racer2"> から登録番号・氏名を取得
    """
    h1 = soup.find("h1", class_="side_title")
    if not h1:
        return {}
    plain = re.sub(r"\s+", " ", h1.get_text(" ", strip=True))
    # 例: "1370 小澤    成吉 オザワ セイキチ"
    m = re.match(r"(?P<regno>\d+)\s+(?P<name>[^\d]+)", plain)
    return m.groupdict() if m else {}


def _table_name(table_tag, index: int) -> str:
    """
    テーブル名を決定
      1. class 属性にユニークっぽい名前があればそれを使う
      2. 直前の見出し (<h3> or <h2>) があればそのテキスト
      3. それでも無ければ "table_{index}"
    """
    classes: List[str] = table_tag.get("class", [])
    if classes:
        return classes[0]
    heading = table_tag.find_previous(["h3", "h2"])
    if heading:
        return re.sub(r"\s+", "_", heading.get_text(strip=True))
    return f"table_{index}"


def html_tables_to_csv(html_path: Path, out_dir: Path) -> None:
    """HTML 内のすべての <table> を DataFrame 化して個別の CSV に保存"""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")

    # レーサー情報を取得して各表に付与
    profile = _racer_profile(soup)

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, table in enumerate(soup.find_all("table"), 1):
        # ヘッダ行
        thead = table.find("thead")
        headers = [_clean(th.get_text()) for th in thead.find_all("th")] if thead else []

        # 本文
        rows = []
        for tr in table.find_all("tr"):
            # <thead> 内 tr は飛ばす
            if tr.find_parent("thead"):
                continue
            cells = [_clean(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

        if not rows:
            continue

        df = pd.DataFrame(rows, columns=headers if len(headers) == len(rows[0]) else None)

        # 数値に変換できる列はしておく
        df = df.apply(_numericize)

        # レーサー情報を加える
        for key, value in profile.items():
            df.insert(0, key, value)

        filenane = html_path.stem
        csv_path = out_dir / f"{filenane}_{_table_name(table, i)}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✔ Saved: {csv_path.relative_to(out_dir.parent)} (rows={len(df)})")


if __name__ == "__main__":
    import os
    dir_path = os.listdir("download/person_record_html")

    for html_file in dir_path:
        os.makedirs("download/person_record_csv", exist_ok=True)
        html_tables_to_csv(Path("download/person_record_html") / html_file, Path("download/person_record_csv"))
    print("All tables extracted.")
