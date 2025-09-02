# -*- coding: utf-8 -*-
# 使い方: このファイルを実行すると、/mnt/data に CSV を2つ出力します。
#  - boatrace_{jcd}_{date}_{rno}R_entries.csv   … 出走表（選手・成績など）
#  - boatrace_{jcd}_{date}_closing_times.csv    … 1R〜12Rの締切予定時刻
#
# 対象HTML: /mnt/data/wakamatsu_racelist_20_20250808_1.html

import re
import os
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd

def txt(x):
    return (x or "").strip()

def to_float(x):
    s = txt(x).replace('%', '').replace('．', '.').replace('−', '-')
    if s in ['', '-', '–', '—', '―', '・', '―.-', '－', '–', '—']:
        return None
    try:
        return float(s)
    except ValueError:
        # STの".16"のように先頭に0がない小数も吸収
        try:
            return float('0' + s) if s.startswith('.') else float(s)
        except:
            return None

def to_int(x):
    s = re.sub(r'[^\d\-]', '', txt(x))
    if s in ['', '-']:
        return None
    try:
        return int(s)
    except:
        return None

def parse_hd_date(soup, html_path=None):
    # hd=YYYYMMDD を最初に見つけたものから
    a = soup.find('a', href=re.compile(r'hd=\d{8}'))
    if not a:
        # ファイル名から推測
        m = re.search(r'(\d{8})', os.path.basename(html_path))
        if m:
            ymd = m.group(1)
        else:
            return None, None
    else:
        m = re.search(r'hd=(\d{8})', a['href'])
        ymd = m.group(1) if m else None

    date_iso = None
    if ymd:
        try:
            date_iso = datetime.strptime(ymd, "%Y%m%d").date().isoformat()
        except:
            pass
    return ymd, date_iso

def parse_rno_and_jcd(soup, html_path=None):
    # タブ(オッズ/直前情報など)のリンクから rno/jcd を取得
    a = soup.find('a', href=re.compile(r'(odds3t|beforeinfo|pcexpect|myexpect|raceresult)\?rno=\d+.*jcd=\d+.*hd=\d{8}'))
    rno = jcd = None
    if a:
        m1 = re.search(r'rno=(\d+)', a['href'])
        m2 = re.search(r'jcd=(\d+)', a['href'])
        rno = int(m1.group(1)) if m1 else None
        jcd = int(m2.group(1)) if m2 else None
    else:
        # ファイル名から推測
        m = re.search(r'_(\d{2})_(\d{8})_(\d+)\.html$', os.path.basename(html_path))
        if m:
            jcd = int(m.group(1))
            rno = int(m.group(3))
    return rno, jcd

def parse_meta(soup, html_path=None):
    title = txt(soup.select_one('.heading2_titleName').get_text()) if soup.select_one('.heading2_titleName') else ''
    place = ''
    img = soup.select_one('.heading2_area img')
    if img and img.has_attr('alt'):
        place = txt(img['alt'])
    # 節/距離
    h3 = soup.select_one('.title16__add2020 h3') or soup.select_one('.title16_titleDetail__add2020')
    stage = ''
    distance_m = None
    if h3:
        h3_text = ' '.join(list(h3.stripped_strings))
        stage = txt(re.sub(r'\s+', ' ', h3_text))
        m = re.search(r'(\d+)\s*m', h3_text)
        if m:
            distance_m = int(m.group(1))
    # 日付
    hd_raw, date_iso = parse_hd_date(soup, html_path=html_path)
    # RNo/JCD
    rno, jcd = parse_rno_and_jcd(soup, html_path=html_path)
    # タブの日付表示（「8月8日 ３日目」等）も拾っておく
    tab_active = soup.select_one('.tab2 li.is-active2 .tab2_inner')
    day_label = ''
    if tab_active:
        day_label = txt(' '.join(list(tab_active.stripped_strings)))
    return {
        'title': title,
        'place': place,
        'stage_text': stage,     # 例: "予選 1800m"
        'distance_m': distance_m,
        'date_iso': date_iso,
        'date_hd': hd_raw,       # 例: "20250808"
        'rno': rno,
        'jcd': jcd,
        'day_label': day_label   # 例: "8月8日 ３日目"
    }

def parse_closing_times(soup, meta):
    # 画面上部のテーブル(締切予定時刻)
    tables = soup.select('div.table1 table')
    if not tables:
        return pd.DataFrame()
    t = tables[0]
    # ヘッダから 1R..12R を抽出（最初の2つは「レース」「(空)」）
    ths = t.select('thead th')
    headers = [txt(th.get_text()) for th in ths]
    # 1R開始のインデックスを探す
    start_idx = None
    for i, h in enumerate(headers):
        if re.fullmatch(r'\d+R', h):
            start_idx = i
            break
    race_headers = headers[start_idx:] if start_idx is not None else []
    # 本文1行目（締切予定時刻の行）を取得（最初の2つの列は見出し）
    tds = t.select_one('tbody tr').find_all('td')
    # 「締切予定時刻」や空白2列をスキップ
    # "レース"列+空列を除いた本当の時刻部分だけ抽出
    # 先頭2列が見出しのケースに対応
    time_cells = [txt(td.get_text()) for td in tds if re.search(r'\d{1,2}:\d{2}', txt(td.get_text())) or txt(td.get_text()) == '']
    # マッピング
    data = []
    for i, name in enumerate(race_headers):
        if not re.fullmatch(r'\d+R', name):
            continue
        time_val = time_cells[i] if i < len(time_cells) else ''
        data.append({
            'date': meta['date_iso'],
            'jcd': meta['jcd'],
            'place': meta['place'],
            'title': meta['title'],
            'race_no': name,
            'closing_time': time_val
        })
    return pd.DataFrame(data)

def expand_day_slots(main_table):
    # 成績ブロックのヘッダ(3行目)から「初日/２日目/...」のラベルを取る（各2枠）
    thead_rows = main_table.select('thead tr')
    if len(thead_rows) < 3:
        # 万一欠けていたら 14コ分のスロットだけ作る
        return [f'slot{i+1}' for i in range(14)], 14
    day_ths = thead_rows[2].find_all('th')
    day_labels = []
    for th in day_ths:
        colsp = int(th.get('colspan', '1'))
        label = txt(th.get_text())
        if not label:
            label = 'その他'
        # 各日2コずつ(= colsp)のスロット名に展開
        for k in range(colsp):
            suffix = k + 1
            day_labels.append(f'{label}_{suffix}')
    return day_labels, len(day_labels)

def parse_entries(soup, meta):
    main_table = soup.select_one('div.table1.is-tableFixed__3rdadd table')
    if not main_table:
        return pd.DataFrame()

    day_slots, slot_count = expand_day_slots(main_table)

    tbodies = main_table.find_all('tbody')
    rows_out = []

    for tb in tbodies:
        trs = tb.find_all('tr', recursive=False)
        if len(trs) != 4:
            # 想定外構造はスキップ
            continue

        # --- Row 0: 静的情報 + レースNo（艇番色） + 早見 ---
        tds0 = trs[0].find_all('td', recursive=False)

        # まず最初の9つのrowspanセルを特定（枠, 写真, 選手, FLST, 全国, 当地, モーター, ボート, スペーサ）
        rowspan_cells = []
        non_rowspan_idx = None
        for i, td in enumerate(tds0):
            if td.has_attr('rowspan'):
                rowspan_cells.append(td)
            else:
                non_rowspan_idx = i
                break

        # 念のため不足を補う（期待は9）
        while len(rowspan_cells) < 9 and non_rowspan_idx is not None:
            # 予期せぬ構造でも落ちないように
            rowspan_cells.append(tds0[len(rowspan_cells)])

        # 枠
        lane_cell = rowspan_cells[0] if len(rowspan_cells) > 0 else None
        lane = to_int(lane_cell.get_text()) if lane_cell else None

        # 写真/プロフィール
        photo_cell = rowspan_cells[1] if len(rowspan_cells) > 1 else None
        photo_url = ''
        profile_url = ''
        if photo_cell:
            img = photo_cell.find('img')
            if img and img.has_attr('src'):
                photo_url = img['src']
            a = photo_cell.find('a')
            if a and a.has_attr('href'):
                profile_url = a['href']

        # 選手情報
        racer_cell = rowspan_cells[2] if len(rowspan_cells) > 2 else None
        reg_no = None
        grade = ''
        name = ''
        branch = ''
        birthplace = ''
        age = None
        weight = None
        if racer_cell:
            # 登録番号/級別
            divs = racer_cell.select('div.is-fs11')
            if divs:
                reg_grade_text = ' '.join(list(divs[0].stripped_strings))
                m = re.search(r'(\d+)\s*/\s*([A|B]\d)', reg_grade_text)
                if m:
                    reg_no = to_int(m.group(1))
                    grade = m.group(2)
            # 氏名
            name_a = racer_cell.select_one('.is-fBold a')
            if name_a:
                name = txt(name_a.get_text())
            # 支部/出身地 & 年齢/体重
            if len(divs) >= 2:
                lines = list(divs[-1].stripped_strings)
                # 期待: [ "福岡/静岡", "26歳/53.0kg" ]
                if lines:
                    # 支部/出身
                    if '/' in lines[0]:
                        br, bp = lines[0].split('/', 1)
                        branch = txt(br)
                        birthplace = txt(bp)
                    # 年齢/体重
                    if len(lines) > 1 and '/' in lines[1]:
                        a_txt, w_txt = lines[1].split('/', 1)
                        age = to_int(a_txt)
                        w_m = re.search(r'([\d\.]+)', w_txt)
                        if w_m:
                            weight = float(w_m.group(1))

        # F/L/ST
        flst_cell = rowspan_cells[3] if len(rowspan_cells) > 3 else None
        f_count = l_count = None
        avg_st = None
        if flst_cell:
            parts = list(flst_cell.stripped_strings)  # ["F0", "L0", "0.18"]
            if len(parts) >= 1:
                f_count = to_int(parts[0].replace('F', ''))
            if len(parts) >= 2:
                l_count = to_int(parts[1].replace('L', ''))
            if len(parts) >= 3:
                avg_st = to_float(parts[2])

        # 全国 / 当地
        def parse_three_lines(td):
            li = list(td.stripped_strings)
            if len(li) >= 3:
                return to_float(li[0]), to_float(li[1]), to_float(li[2])
            return None, None, None

        national_win, national_2r, national_3r = (None, None, None)
        local_win, local_2r, local_3r = (None, None, None)

        if len(rowspan_cells) > 4 and rowspan_cells[4]:
            national_win, national_2r, national_3r = parse_three_lines(rowspan_cells[4])
        if len(rowspan_cells) > 5 and rowspan_cells[5]:
            local_win, local_2r, local_3r = parse_three_lines(rowspan_cells[5])

        # モーター / ボート
        motor_no = motor_2r = motor_3r = None
        boat_no = boat_2r = boat_3r = None
        if len(rowspan_cells) > 6 and rowspan_cells[6]:
            li = list(rowspan_cells[6].stripped_strings)
            if len(li) >= 1: motor_no = to_int(li[0])
            if len(li) >= 2: motor_2r = to_float(li[1])
            if len(li) >= 3: motor_3r = to_float(li[2])
        if len(rowspan_cells) > 7 and rowspan_cells[7]:
            li = list(rowspan_cells[7].stripped_strings)
            if len(li) >= 1: boat_no = to_int(li[0])
            if len(li) >= 2: boat_2r = to_float(li[1])
            if len(li) >= 3: boat_3r = to_float(li[2])

        # レースNo（艇番色）
        # non_rowspan_idx は最初の成績ブロックの列の開始位置
        if non_rowspan_idx is None:
            # 安全装置
            non_rowspan_idx = 9
        # 0行目の成績セル
        race_no_cells = tds0[non_rowspan_idx:non_rowspan_idx + slot_count]
        # 早見
        hayami_cell = None
        if len(tds0) > non_rowspan_idx + slot_count:
            hayami_cell = tds0[non_rowspan_idx + slot_count]
        hayami_text = ''
        hayami_href = ''
        if hayami_cell:
            a = hayami_cell.find('a')
            if a:
                hayami_text = txt(a.get_text())
                hayami_href = a.get('href', '')

        # --- Row 1: 進入コース ---
        tds1 = trs[1].find_all('td', recursive=False)
        course_cells = tds1[:slot_count] if len(tds1) >= slot_count else tds1

        # --- Row 2: STタイミング ---
        tds2 = trs[2].find_all('td', recursive=False)
        st_cells = tds2[:slot_count] if len(tds2) >= slot_count else tds2

        # --- Row 3: 成績 ---
        tds3 = trs[3].find_all('td', recursive=False)
        result_cells = tds3[:slot_count] if len(tds3) >= slot_count else tds3

        # 行データ作成
        row = {
            'date': meta['date_iso'],
            'jcd': meta['jcd'],
            'place': meta['place'],
            'title': meta['title'],
            'day_label': meta['day_label'],
            'distance_m': meta['distance_m'],
            'rno': meta['rno'],
            'lane': lane,
            'reg_no': reg_no,
            'grade': grade,
            'name': name,
            'branch': branch,
            'birthplace': birthplace,
            'age': age,
            'weight_kg': weight,
            'F_count': f_count,
            'L_count': l_count,
            'avg_ST': avg_st,
            'national_win': national_win,
            'national_2ren': national_2r,
            'national_3ren': national_3r,
            'local_win': local_win,
            'local_2ren': local_2r,
            'local_3ren': local_3r,
            'motor_no': motor_no,
            'motor_2ren': motor_2r,
            'motor_3ren': motor_3r,
            'boat_no': boat_no,
            'boat_2ren': boat_2r,
            'boat_3ren': boat_3r,
            'photo_url': photo_url,
            'profile_url': profile_url,
            'hayami': hayami_text,
            'hayami_href': hayami_href,
        }

        # スロットごとの「レースNo（艇番色） / 進入 / ST / 成績」
        for idx in range(slot_count):
            slot_name = day_slots[idx]

            # レースNo + 色
            rn_td = race_no_cells[idx] if idx < len(race_no_cells) else None
            rn_text = txt(rn_td.get_text()) if rn_td else ''
            # 色番号（is-boatColorX から抽出）
            color_num = ''
            if rn_td:
                classes = rn_td.get('class', [])
                for c in classes:
                    m = re.match(r'is-boatColor(\d)', c)
                    if m:
                        color_num = m.group(1)
                        break

            # 進入
            course_td = course_cells[idx] if idx < len(course_cells) else None
            course_text = txt(course_td.get_text()) if course_td else ''

            # ST
            st_td = st_cells[idx] if idx < len(st_cells) else None
            st_val = to_float(st_td.get_text()) if st_td else None

            # 成績
            res_td = result_cells[idx] if idx < len(result_cells) else None
            # 成績はリンク内テキストがあればそれを
            res_text = ''
            if res_td:
                a = res_td.find('a')
                res_text = txt(a.get_text()) if a else txt(res_td.get_text())

            row[f'{slot_name}_race_no'] = rn_text
            row[f'{slot_name}_lane_color'] = color_num
            row[f'{slot_name}_course'] = course_text
            row[f'{slot_name}_ST'] = st_val
            row[f'{slot_name}_result'] = res_text

        rows_out.append(row)

    df = pd.DataFrame(rows_out)

    # 見やすさのため並び替え（任意）
    base_cols = [
        'date', 'jcd', 'place', 'title', 'day_label', 'distance_m', 'rno',
        'lane', 'reg_no', 'grade', 'name', 'branch', 'birthplace', 'age', 'weight_kg',
        'F_count', 'L_count', 'avg_ST',
        'national_win', 'national_2ren', 'national_3ren',
        'local_win', 'local_2ren', 'local_3ren',
        'motor_no', 'motor_2ren', 'motor_3ren',
        'boat_no', 'boat_2ren', 'boat_3ren',
        'photo_url', 'profile_url', 'hayami', 'hayami_href'
    ]
    slot_cols = []
    for s in day_slots:
        slot_cols += [f'{s}_race_no', f'{s}_lane_color', f'{s}_course', f'{s}_ST', f'{s}_result']

    cols = [c for c in base_cols if c in df.columns] + [c for c in slot_cols if c in df.columns]
    df = df.reindex(columns=cols)
    return df

def main(is_pred=False):
    html_dir_name = 'download/wakamatsu_off_racelist_html'
    csv_dir_name = 'download/wakamatsu_off_racelist_csv'
    if is_pred:
        html_dir_name = 'download/wakamatsu_off_racelist_pred_html'
        csv_dir_name = 'download/wakamatsu_off_racelist_pred_csv'
    for html_filename in os.listdir(html_dir_name):
        if not html_filename.endswith('.html'):
            continue
        html_path = os.path.join(html_dir_name, html_filename)
        print(f"Parsing HTML: {html_path}")

        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'lxml')

        meta = parse_meta(soup)

        # 出走表
        entries_df = parse_entries(soup, meta)

        # 締切予定時刻
        closing_df = parse_closing_times(soup, meta)

        # 出力ファイル名
        csv_filename = html_filename.replace('.html', f'.csv')
        csv_path = os.path.join(csv_dir_name, csv_filename)
        out_entries = csv_path.replace('.csv', f'_entries.csv')
        out_closing = csv_path.replace('.csv', f'_closing_times.csv')

        entries_df.to_csv(out_entries, index=False, encoding='utf-8-sig')
        closing_df.to_csv(out_closing, index=False, encoding='utf-8-sig')

        print(f"出走表CSV: {out_entries}  行数={len(entries_df)}")
        print(f"締切予定時刻CSV: {out_closing}  行数={len(closing_df)}")

if __name__ == "__main__":
    main()
