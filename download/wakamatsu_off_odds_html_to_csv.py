# -*- coding: utf-8 -*-
"""
boat_race_3rentan_odds_to_csv.py

若松 2024-01-01 1R の 3連単オッズページ（例: odds3t.html）をパースし、
racers.csv / odds_matrix.csv / trifecta_long.csv を出力する。
"""

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

def parse_wakamatsu_odds_html(html_path: str, enc: str = "utf-8") -> None:
    src = Path(html_path)
    print(f"Parsing HTML: {src}")
    csv_dir = "download/wakamatsu_off_odds3t_csv"
    out_dir = Path(csv_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    # ------------------------------------------------------------
    # 1. HTML 読み込み
    # ------------------------------------------------------------
    with src.open(encoding=enc) as f:
        soup = BeautifulSoup(f, "html.parser")

    main_label = soup.select_one(".heading1_mainLabel").get_text(strip=True) if soup.select_one(".heading1_mainLabel") else None
    if main_label and "データがありません" in main_label:
        print("⚠️ データがありません。スキップします。")
        return
    if main_label and "エラー" in main_label:
        # エラーcsvを出力
        out_path = out_dir / f"{src.stem}_error.csv"
        with out_path.open("w", encoding=enc) as f:
            f.write(f"error_message,{main_label}\n")
        print(f"⚠️ エラー情報を {out_path} に保存しました")
        return
    
    race_date  = soup.select_one(".tab2_tabs li.is-active2").get_text(strip=True)  # 例: '1月1日４日目'

    if race_date and "中止" in race_date:
        csv_dir = "download/wakamatsu_off_beforeinfo_csv"
        Path(csv_dir).mkdir(exist_ok=True)
        print(f"中止レースのメタデータを保存: {src.stem}_meta.csv")
        meta_df = pd.DataFrame([{
            "date_label": race_date,
        }])
        meta_df.to_csv(f"{csv_dir}/{src.stem}_meta.csv", index=False)
        return

    cancel_tag = soup.select_one(".title12_title.is-type1")
    if cancel_tag:                           # ← 中止レース
        notice = cancel_tag.get_text(strip=True)
        print("中止通知:", notice)
        return

    # ------------------------------------------------------------
    # 2. ヘッダ行から『艇番⇔選手名』対応表を作成
    # ------------------------------------------------------------
    thead = soup.select_one(".title7 + div.table1 thead")
    racers = []
    ths = thead.find_all("th")

    for i in range(0, len(ths), 2):
        num_txt  = ths[i].get_text(strip=True)
        name_txt = ths[i+1].get_text(strip=True)

        # 数字セルかどうかを確認してから変換
        if num_txt.isdigit():
            racers.append({
                "boat_no":   int(num_txt),
                "racer_name": name_txt
            })                # [番号, 名前セル(colspan=2)]×6 = 18 th 要素

    # ------------------------------------------------------------
    # 3. オッズ本体テーブルを DataFrame 化（行列そのまま）
    #    - rowspan, colspan を解決するため pandas.read_html を利用
    #    - 欠損を前方／下方向で埋める → human readable な行列として保存
    # ------------------------------------------------------------
    table_html = str(soup.select_one(".title7 + div.table1 table"))
    matrix_df  = pd.read_html(table_html, flavor="lxml")[0]

    # 欠損補完: 左→右 に流し、その後 上→下 に流す
    matrix_df = matrix_df.ffill(axis=1).ffill()

    # 保存（そのままでは行数が多いのでインデックス付きで melt してもよい）
    # ------------------------------------------------------------
    # 4. 『first, second, third, odds』長形式へ変換（ベストエフォート）
    #    テーブルの構造ルール:
    #       - 各 <tr> 先頭の rowspan セル (class に is-borderLeftNone) …「first 頭」(A)
    #       - 以降、3セルで 1グループ: [rowspan セル (second=B), numeric (third=C), odds]
    #       - ただし B==A のグループも存在し得る。重複回避のため A,B,C がすべて異なるもののみ採用
    # ------------------------------------------------------------
    long = []
    for tr in soup.select(".title7 + div.table1 tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        # グループ先頭 (TD0) が「first」
        first = tds[0].text.strip()
        first = int(first) if first.isdigit() else None
        if first is None:
            continue

        # 以降 3セルずつで second, third, odds
        group = tds[1:]                # TD0 を除く
        for j in range(0, len(group), 3):
            if j + 2 >= len(group):
                break
            second_td, third_td, odds_td = group[j : j+3]

            # second: rowspan セル
            second = second_td.text.strip()
            second = int(second) if second.isdigit() else None

            # third: 通常セル
            third  = third_td.text.strip()
            third  = int(third) if third.isdigit() else None

            # odds
            odds_txt = odds_td.text.strip().replace(",", "")
            odds     = None
            if re.fullmatch(r"\d+(\.\d+)?", odds_txt):
                odds = float(odds_txt)

            # バリデーション: 3数値揃い & 重複なし
            if None not in (first, second, third, odds) and \
            len({first, second, third}) == 3:
                long.append({
                    "first":  first,
                    "second": second,
                    "third":  third,
                    "odds":   odds
                })

    out_path = out_dir / f"{src.stem}"
    pd.DataFrame(racers).to_csv(f"{out_path}_racers.csv",
                                index=False, encoding="utf-8-sig")
    matrix_df.to_csv(f"{out_path}_odds_matrix.csv", index=False, encoding="utf-8-sig")
    long_df = pd.DataFrame(long)
    long_df.to_csv(f"{out_path}_trifecta_long.csv", index=False, encoding="utf-8-sig")

    print("✅ racers.csv / odds_matrix.csv / trifecta_long.csv を出力しました")

if __name__ == "__main__":
    import os
    htmls = os.listdir("download/wakamatsu_off_odds3t_html")
    for html in htmls:
        parse_wakamatsu_odds_html(f"download/wakamatsu_off_odds3t_html/{html}", enc="utf-8")
