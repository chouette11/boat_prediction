#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_boat_html.py

若松『西部発刊２５周年記念スポーツ報知杯年またぎ特選競走』HTML から各種情報を抽出し、
CSV に保存するスクリプト。使い方:

    python parse_boat_html.py beforeinfo.html
"""
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def parse_boat_race_html(html_content: str, encoding: str = "utf-8", is_pred: bool = False) -> None:
    """BOAT RACE の『直前情報』ページ（PC 向け）からデータを抽出し CSV を生成する"""
    print(html_content[:100])  # HTMLの一部を表示して確認
    soup = BeautifulSoup(html_content, "html.parser")
    print(f"soup.title: {soup.title.get_text(strip=True)}")

    main_label = soup.select_one(".heading1_mainLabel").get_text(strip=True) if soup.select_one(".heading1_mainLabel") else None
    if main_label and "データがありません" in main_label or main_label and "エラー" in main_label:
        print("⚠️ データがありません。スキップします。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ------------------------------------------------------------------
    # 1) 開催場・レース名・日付
    # ------------------------------------------------------------------
    race_title = soup.select_one(".heading2_titleName").get_text(strip=True)
    race_place = soup.select_one(".heading2_area img")["alt"]          # 例: 若松
    if len(race_place) == 2:
        #半角スペースを二文字の間に挿入
        race_place = f"{race_place[0]} {race_place[1]}"
    race_date  = soup.select_one(".tab2_tabs li.is-active2").get_text(strip=True)  # 例: '1月1日４日目'

    if race_date and "中止" in race_date:
        csv_dir = "download/wakamatsu_off_beforeinfo_csv"
        Path(csv_dir).mkdir(exist_ok=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    meta_df = pd.DataFrame([{
        "place": race_place,
        "race_title": race_title,
        "date_label": race_date,
    }])

    # ------------------------------------------------------------------
    # 3) 出走表（選手情報）
    # ------------------------------------------------------------------
    racers_data = []
    for tbody in soup.select("table.is-w748 tbody.is-fs12"):
        lane_td = tbody.find("td", class_=re.compile(r"is-boatColor"))
        if lane_td is None:  # 部品交換凡例など別 tbody はスキップ
            continue

        lane = lane_td.get_text(strip=True)            # 枠番
        anchor = tbody.select_one('a[href*="profile?toban="]')
        name   = anchor.get_text(strip=True)
        racer_id = re.search(r"toban=(\d+)", anchor["href"]).group(1)

        photo_src = tbody.select_one("img")["src"]

        # 体重・調整重量
        weight_cell = tbody.find("td", string=re.compile(r"kg"))
        weight = weight_cell.get_text(strip=True) if weight_cell else ""
        adj_w_cell = weight_cell.find_next("td", string=re.compile(r"^\d+\.\d$")) if weight_cell else None
        adj_weight = adj_w_cell.get_text(strip=True) if adj_w_cell else ""

        # 展示タイム・チルト
        ex_time_cell = tbody.find("td", string=re.compile(r"^\d+\.\d{2}$")) if tbody else None
        exhibition_time = ex_time_cell.get_text(strip=True) if ex_time_cell else ""
        tilt_cell = ex_time_cell.find_next("td") if ex_time_cell else None
        tilt = tilt_cell.get_text(strip=True) if tilt_cell else ""

        racers_data.append({
            "lane": lane,
            "racer_id": racer_id,
            "name": name,
            "weight": weight,
            "adjust_weight": adj_weight,
            "exhibition_time": exhibition_time,
            "tilt": tilt,
            "photo": photo_src,
        })

    racers_df = pd.DataFrame(racers_data).sort_values("lane")

    # ------------------------------------------------------------------
    # 4) スタート展示（ST）
    # ------------------------------------------------------------------
    st_data = []
    for i, div in enumerate(soup.select(".table1_boatImage1")):
        lane = div.select_one(".table1_boatImage1Number").get_text(strip=True)
        course = i + 1  # 進入
        st   = div.select_one(".table1_boatImage1Time").get_text(strip=True)
        st_data.append({"lane": lane, "ST": st, "course": course})

    if not st_data:
        csv_dir = "download/wakamatsu_off_beforeinfo_csv"
        Path(csv_dir).mkdir(exist_ok=True)
        meta_df = pd.DataFrame([{
            "place": race_place,
            "race_title": race_title,
            "date_label": race_date,
        }])
        return pd.DataFrame(), pd.DataFrame(), meta_df


    start_ex_df = pd.DataFrame(st_data).sort_values("lane")

    racers_df = racers_df.merge(start_ex_df, on="lane", how="left")
    # ------------------------------------------------------------------
    # 5) 水面気象
    # ------------------------------------------------------------------
    wblock = soup.select_one(".weather1")
    weather_df = pd.DataFrame([{
        "obs_datetime_label": wblock.select_one(".weather1_title").get_text(strip=True),
        "weather": wblock.select_one(".is-weather .weather1_bodyUnitLabelTitle").get_text(strip=True),
        "air_temp_C": wblock.select_one(".is-direction .weather1_bodyUnitLabelData").get_text(strip=True),
        "wind_speed_m": wblock.select_one(".is-wind .weather1_bodyUnitLabelData").get_text(strip=True),
        "water_temp_C": wblock.select_one(".is-waterTemperature .weather1_bodyUnitLabelData").get_text(strip=True),
        "wave_height_cm": wblock.select_one(".is-wave .weather1_bodyUnitLabelData").get_text(strip=True),
        # 風向アイコンの class → 'is-wind4' 等。数値は方角コードに対応
        "wind_dir_icon": wblock.select_one(".is-windDirection p")["class"][-1],
    }])

    return racers_df, weather_df, meta_df

if __name__ == "__main__":
    import os
    # htmlのファイルを取得
    dir_name = "download/wakamatsu_off_beforeinfo_html_2223"
    htmls = os.listdir(dir_name)
    for html in htmls:
        parse_boat_race_html(f"{dir_name}/{html}", encoding="utf-8")