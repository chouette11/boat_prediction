#!/usr/bin/env python3
"""
parse_boatrace_html.py

HTML 出走表（質問の例を想定）を解析し、レーサーデータを CSV に保存する
"""
import csv
import re
from pathlib import Path
from typing import List, Dict

from bs4 import BeautifulSoup


def _num(text: str, _type=float):
    """数字だけ取り出し型変換。失敗時は None を返す"""
    m = re.search(r"[-+]?\d*\.?\d+", text)
    try:
        return _type(m.group()) if m else None
    except ValueError:
        return None


def parse_html(html: str) -> List[Dict]:
    """HTML 文字列を解析してレコードの list[dict] を返す"""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select("div.tblRL > table")
    for table in tables:
        if table is None:
            raise ValueError("出走表テーブル (div.tblRL) が見つかりません")

        # tr を最上位だけで取得（ネストした table 内の tr までは入れない）
        rows = [tr for tr in table.find_all("tr", recursive=False)]

        def _get_counts(cell):
            nums = [int(t.text) for t in cell.select("span.tdRitsu2Text")]
            starts = int(cell.select_one("td.tdRitsu2Text2").text)
            return (*nums, starts)  # 1st, 2nd, 3rd, starts

        def _get_counts2(cell):
            nums = [int(t.text) for t in cell.select("td.tdMBsum1, td.tdMBsum2")]
            starts = int(cell.select_one("td.tdMBsum3").text)
            return (*nums, starts)

        # 各列 = 号艇 1〜6 → インデックス 1..6
        result = []
        for col in range(1, 7):
            # 各行から該当列の td を抜き出す（行によっては <td> が無い事もある）
            cells = []
            for r in rows:
                tds = r.find_all("td", recursive=False)
                if len(tds) > col:
                    cells.append(tds[col])
                else:
                    cells.append(None)  # プレースホルダ

            # 基本情報 --------------------------
            reg_no = _num(cells[2].get_text(), int)  # 登録番号
            name_age_txt = cells[3].get_text(" ", strip=True)
            name_match = re.match(r"(.+?)\s*\((\d+)\)", name_age_txt)
            name = name_match.group(1).strip() if name_match else name_age_txt
            age = int(name_match.group(2)) if name_match else None

            # 級別 ------------------------------
            class_hist_str = " ".join(
                [td.get_text(strip=True) for td in cells[4].select("td")]
            )
            class_hist = class_hist_str.split(" ")

            # 能力指数 --------------------------
            ability_now = cells[5].font.text if cells[5].font else None
            ability_prev = None

            # F / L ----------------------------
            f_now = _num(cells[6].select_one(".jF").text, int)
            l_now = _num(cells[6].select_one(".jL").text, int)

            # 全国成績 --------------------------
            winrate_natl = _num(cells[7].select_one(".tdRitsuWin").text, float)
            twoin_natl = _num(cells[7].select("span.tdRitsuR")[0].text, float)
            threein_natl = _num(cells[7].select("span.tdRitsuR")[1].text, float)

            nat_counts_cell = cells[8]   # 「1|2|3着数 出走数 (6ヶ月)」
            loc_counts_cell = cells[11]  # 「1|2|3着数 出走数」(当地)

            nat_1st, nat_2nd, nat_3rd, nat_starts = _get_counts(nat_counts_cell)
            loc_1st, loc_2nd, loc_3rd, loc_starts = _get_counts(loc_counts_cell)
            
            # モーター --------------------------
            motor_block = cells[13]
            motor_no = _num(motor_block.select_one(".tdMBno").text, int)
            motor_rates = motor_block.select("b")
            motor_2in = _num(motor_rates[0].text, float) if motor_rates else None
            motor_3in = _num(motor_rates[1].text, float) if motor_rates else None
            print(_get_counts2(cells[14]))
            mot_1, mot_2, mot_3, mot_st = _get_counts2(cells[14])

            # ボート ----------------------------
            boat_block = cells[15]
            boat_no_hw = _num(boat_block.select_one(".tdMBno").text, int)
            boat_rates = boat_block.select("b")
            boat_2in = _num(boat_rates[0].text, float) if boat_rates else None
            boat_3in = _num(boat_rates[1].text, float) if boat_rates else None
            print(cells[16])
            boa_1, boa_2, boa_3, boa_st = _get_counts2(cells[16])

            result.append(
                {
                    "boat_no": col,
                    "reg_no": reg_no,
                    "name": name,
                    "age": age,
                    "class_now": class_hist[0],
                    "class_hist1": class_hist[1] if len(class_hist) > 1 else "",
                    "class_hist2": class_hist[2] if len(class_hist) > 2 else "",
                    "class_hist3": class_hist[3] if len(class_hist) > 3 else "",
                    "ability_now": ability_now,
                    "ability_prev": ability_prev,
                    "F_now": f_now,
                    "L_now": l_now,
                    "winrate_natl": winrate_natl,
                    "2in_natl": twoin_natl,
                    "3in_natl": threein_natl,
                    "nat_1st": nat_1st,
                    "nat_2nd": nat_2nd,
                    "nat_3rd": nat_3rd,
                    "nat_starts": nat_starts,
                    "loc_1st": loc_1st,
                    "loc_2nd": loc_2nd,
                    "loc_3rd": loc_3rd,
                    "loc_starts": loc_starts,
                    "motor_no": motor_no,
                    "motor_2in": motor_2in,
                    "motor_3in": motor_3in,
                    "mot_1st": mot_1,
                    "mot_2nd": mot_2,
                    "mot_3rd": mot_3,
                    "mot_starts": mot_st,
                    "boat_no_hw": boat_no_hw,
                    "boat_2in": boat_2in,
                    "boat_3in": boat_3in,
                    "boa_1st": boa_1,
                    "boa_2nd": boa_2,
                    "boa_3rd": boa_3,
                    "boa_starts": boa_st,
                }
            )
        return result


def save_csv(records: List[Dict], out_path: Path) -> None:
    """レコードリストを CSV ファイルへ書き出し"""
    if not records:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="競艇 HTML 出走表を CSV に変換", prog="parse_boatrace_html"
    )
    parser.add_argument("html_file", help="入力 HTML ファイル")
    parser.add_argument(
        "-o", "--output", default="boatrace_output.csv", help="出力 CSV パス"
    )
    args = parser.parse_args()

    html_text = Path(args.html_file).read_text(encoding="utf-8", errors="ignore")
    recs = parse_html(html_text)
    save_csv(recs, Path(args.output))
    print(f"{len(recs)} records written to {args.output}", file=sys.stderr)
