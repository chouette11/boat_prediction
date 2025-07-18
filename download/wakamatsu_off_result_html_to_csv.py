from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from bs4 import BeautifulSoup

def tidy(text: str | None) -> str:
    "Collapse whitespace and strip."
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_race_no(soup: BeautifulSoup) -> str:
    """Try to infer race number (1R, 2R …) from the first link that
    contains 'rno=' inside .tab3_tabs."""
    link = soup.select_one(".tab3_tabs a[href*='rno=']")
    if not link:
        return ""
    href = link["href"]
    qs = parse_qs(urlparse(href).query)
    return qs.get("rno", [""])[0]


def parse_results_table(soup: BeautifulSoup) -> list[dict[str, str]]:
    "Return list of dictionaries for each finisher."
    # Identify the first table whose header begins with '着'
    table = None
    for t in soup.select("table.is-w495"):
        th = tidy(t.select_one("thead th").get_text()) if t.select_one("thead th") else ""
        if th == "着":
            table = t
            break
    if table is None:
        return []

    # Map lane → ST & tactic from the start info table (if present)
    st_map: dict[str, tuple[str, str]] = {}
    for i, unit in enumerate(soup.select(".table1_boatImage1")):
        lane = tidy((unit.select_one(".table1_boatImage1Number") or {}).get_text())
        txt = tidy((unit.select_one(".table1_boatImage1TimeInner") or {}).get_text())
        entry = i + 1
        print(f"Lane {lane} ST: {txt} (entry {entry})")
        if not lane:
            continue
        parts = txt.split()
        st_time = parts[0] if parts else ""
        tactic = parts[1] if len(parts) >= 2 else ""
        print(f"Lane {lane} ST: {st_time} Tactic: {tactic}")
        st_map[lane] = (entry, st_time, tactic)

    rows: list[dict[str, str]] = []

    for tbody in table.select("tbody"):
        td = tbody.select_one("tr")
        if not td:
            continue
        cells = td.find_all("td")
        if len(cells) < 4:
            continue
        position = tidy(cells[0].get_text())
        lane = tidy(cells[1].get_text())
        racer_cell = cells[2]
        spans = racer_cell.find_all("span")
        racer_no = tidy(spans[0].get_text()) if spans else ""
        racer_name = tidy(spans[1].get_text()) if len(spans) > 1 else tidy(racer_cell.get_text())
        race_time = tidy(cells[3].get_text())

        print(st_map)
        entry, st_time, tactic = st_map.get(lane, ("", "", ""))
        rows.append(
            {
                "position_txt": position,
                "lane": lane,
                "racer_no": racer_no,
                "racer_name": racer_name,
                "time": race_time,
                "st_entry": entry,
                "st_time": st_time,
                "tactic": tactic,
            }
        )
    return rows


def parse_payouts_table(soup: BeautifulSoup) -> list[dict[str, str]]:
    "Extract bet‑type payout rows."
    payouts: list[dict[str, str]] = []
    # Table whose header begins with '勝式'
    table = None
    for t in soup.select("table.is-w495"):
        th = tidy(t.select_one("thead th").get_text()) if t.select_one("thead th") else ""
        if th == "勝式":
            table = t
            break
    if table is None:
        return payouts

    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        bet_type = tidy(cells[0].get_text())
        # Skip rows where bet_type cell is empty (continuation rows)
        if not bet_type:
            continue
        comb = tidy(cells[1].get_text().replace("\u00a0", " "))
        payout = tidy(cells[2].get_text().replace("¥", "").replace("￥", "").replace(",", ""))
        popularity = tidy(cells[3].get_text())
        payouts.append(
            {
                "bet_type": bet_type,
                "combination": comb,
                "payout_yen": payout,
                "popularity": popularity,
            }
        )
    return payouts


def parse_weather(soup: BeautifulSoup) -> dict[str, str]:
    "Extract simple weather block (optional)."
    w = {}
    spin = lambda sel: tidy((soup.select_one(sel) or {}).get_text())
    w["temperature_c"] = spin(".weather1_bodyUnit.is-direction .weather1_bodyUnitLabelData")
    w["weather"] = spin(".weather1_bodyUnit.is-weather .weather1_bodyUnitLabelTitle")
    w["wind_speed_m"] = spin(".weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData")
    w["wind_dir_class"] = (soup.select_one(".weather1_bodyUnit.is-windDirection p") or {}).get(
        "class", []
    )[-1] if soup.select_one(".weather1_bodyUnit.is-windDirection p") else ""
    w["water_temp_c"] = spin(".weather1_bodyUnit.is-waterTemperature .weather1_bodyUnitLabelData")
    w["wave_height_cm"] = spin(".weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData")
    return {k: v for k, v in w.items() if v}


def main() -> None:
    import os
    htmls = os.listdir("download/wakamatsu_off_raceresult_html")
    csv_dir = Path("download/wakamatsu_off_raceresult_csv")
    csv_dir.mkdir(parents=True, exist_ok=True)
    for html in htmls:
        if not html.endswith(".html"):  
            print(f"Skipping non-HTML file: {html}")
            continue
        input_html = Path(f"download/wakamatsu_off_raceresult_html/{html}")
        print(f"Processing {input_html}...")

        soup = BeautifulSoup(input_html.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        main_label = soup.select_one(".heading1_mainLabel").get_text(strip=True) if soup.select_one(".heading1_mainLabel") else None
        if main_label and "データがありません" in main_label:
            print("⚠️ データがありません。スキップします。")
            continue
        if main_label and "エラー" in main_label:
            # エラーcsvを出力
            out_path = csv_dir / f"{input_html.stem}_error.csv"
            with out_path.open("w", encoding="utf-8-sig") as f:
                f.write(f"error_message,{main_label}\n")
            print(f"⚠️ エラー情報を {out_path} に保存しました")
            continue

        stadium = tidy((soup.select_one(".heading2_area img[alt]") or {}).get("alt", ""))
        race_title = tidy((soup.select_one(".heading2_titleName") or {}).get_text())
        date_label = tidy((soup.select_one(".tab2_tabs li.is-active2") or {}).get_text())
        race_no = get_race_no(soup)

        res_rows = parse_results_table(soup)
        for r in res_rows:
            r.update(
                {
                    "stadium": stadium,
                    "race_title": race_title,
                    "date_label": date_label,
                    "race_no": race_no,
                    "source_file": str(input_html),
                }
            )

        payout_rows = parse_payouts_table(soup)
        for p in payout_rows:
            p.update(
                {
                    "stadium": stadium,
                    "race_title": race_title,
                    "date_label": date_label,
                    "race_no": race_no,
                    "source_file": str(input_html),
                }
            )

        # Write CSVs
        def write_csv(rows: list[dict[str, str]], suffix: str):
            if not rows:
                return
            out = csv_dir / f"{input_html.stem}_{suffix}.csv"
            with out.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {len(rows)} rows → {out.name}")

        write_csv(res_rows, "results")

if __name__ == "__main__":
    main()