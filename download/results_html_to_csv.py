import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def parse_html(html: str) -> pd.DataFrame:
    """若松レース HTML からレースカードを DataFrame 化"""
    soup = BeautifulSoup(html, "html.parser")

    # 日付（<h2>ボートレース 若松 - 2024/01/01 …</h2>）を取得
    header = soup.find("h2")
    m = re.search(r"\d{4}/\d{2}/\d{2}", header.text) if header else None
    race_date = m.group(0).replace("/", "-") if m else ""

    records = []
    # #1R, #2R … #12R を順にたどる
    for rno in range(1, 13):
        anchor = soup.find("a", {"name": f"{rno}R"})
        if not anchor:
            continue
        table = anchor.find_next("div", class_="tblRL")
        if not table:
            continue
        rows = table.find_all("tr")

        # 登録番号行（7セル目までが4桁数字）
        reg_row = next(
            (tr for tr in rows
             if len(tr.find_all("td")) >= 7 and
             all(re.fullmatch(r"\d{4}", td.get_text(strip=True))
                 for td in tr.find_all("td")[1:])), None)
        if not reg_row:
            continue

        # 各行ごとの値を抜き出し
        regs  = [td.get_text(strip=True) for td in reg_row.find_all("td")[1:]]

        name_row  = reg_row.find_next("tr")
        names = [td.get_text(strip=True).split("(")[0]
                 for td in name_row.find_all("td")[1:]]

        grade_row = name_row.find_next("tr")
        grades = []
        for td in grade_row.find_all("td")[1:]:
            span = td.find("span", class_="rankAB")
            grades.append(span.get_text(strip=True) if span else "")

        for lane, (reg, name, grade) in enumerate(zip(regs, names, grades), start=1):
            records.append(
                {"date": race_date,
                 "race_no": rno,
                 "lane": lane,
                 "registration": reg,
                 "racer_name": name,
                 "grade": grade}
            )

    return pd.DataFrame(records)


def main() -> None:
    html_dir = Path("download/wakamatsu_result_html")
    csv_dir = Path("download/wakamatsu_result_csv")
    file_name = "wakamatsu_result_20240101.html"
    html_path = html_dir / file_name 
    csv_path = csv_dir / file_name.replace(".html", ".csv")
    html = Path(html_path).read_text(encoding="utf-8")
    df = parse_html(html)
    out = csv_path or Path(html_path).with_suffix(".csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ CSV 出力完了 → {out}")


if __name__ == "__main__":
    main()
