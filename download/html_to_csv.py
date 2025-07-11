import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import os

odds = os.listdir("odds_html")  # HTMLファイルが保存されているディレクトリ

for html_file in odds:
    html_path = os.path.join("odds_html", html_file)
    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    # 1. HTMLをパース
    soup = BeautifulSoup(html_content, "html.parser")

    # 2. データ格納用リストを初期化
    data = []

    # 3. 全レースのテーブルを取得
    race_tables = soup.select("#raceResult table")

    # 4. テーブルごとに解析
    for table in race_tables:
        header = table.find("th", class_="lbl_txt")
        if not header:
            continue

        # グレード、日付、レース名を取得
        grade_elem = header.select_one(".gradeText")
        grade = grade_elem.get_text(strip=True) if grade_elem else ""

        date_elem = header.select_one(".lbl_date")
        name_elem = header.select_one(".lbl_name a")
        if not date_elem or not name_elem:
            continue

        # 日付を YYYY-MM-DD に変換
        raw_date = date_elem.get_text(strip=True)
        match = pd.Series(raw_date).str.extract(r"(\d{4})年(\d{2})月(\d{2})日")
        date = "-".join(match.iloc[0].dropna()) if not match.empty else ""

        name = name_elem.get_text(strip=True)

        # 各行のデータ（1レース分）
        rows = table.select("tbody tr")
        for row in rows:
            cols = row.select("td")
            if not cols or "stop" in cols[1].get("class", []):
                continue

            try:
                rno = cols[0].get_text(strip=True)
                odds_3tan = cols[2].get_text(strip=True).replace(",", "")
                odds_3tan_pop = cols[3].get_text(strip=True)
                odds_3fuku = cols[5].get_text(strip=True).replace(",", "")
                odds_3fuku_pop = cols[6].get_text(strip=True)
                odds_2tan = cols[8].get_text(strip=True).replace(",", "")
                odds_2tan_pop = cols[9].get_text(strip=True)
                odds_2fuku = cols[11].get_text(strip=True).replace(",", "")
                odds_2fuku_pop = cols[12].get_text(strip=True)
            except IndexError:
                continue

            data.append({
                "grade": grade,
                "date": date,
                "race_name": name,
                "race_no": rno,
                "3連単オッズ": odds_3tan,
                "3連単人気": odds_3tan_pop,
                "3連複オッズ": odds_3fuku,
                "3連複人気": odds_3fuku_pop,
                "2連単オッズ": odds_2tan,
                "2連単人気": odds_2tan_pop,
                "2連複オッズ": odds_2fuku,
                "2連複人気": odds_2fuku_pop
            })

    # 5. pandasで整形
    df = pd.DataFrame(data)

    # 6. CSVとして保存
    csv_file = f"odds_csv/{os.path.splitext(html_file)[0]}.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    print(f"✅ CSVファイルに保存しました: {csv_file}")
