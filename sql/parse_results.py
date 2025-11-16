# parse_results.py
# -*- coding: utf-8 -*-

import argparse
import psycopg2
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import sys

BET_TYPES = [
    '単勝','複勝',
    '拡連複','ワイド',
    '二連単','２連単','2連単',
    '二連複','２連複','2連複',
    '三連単','３連単','3連単',
    '三連複','３連複','3連複'
]
WIN_METHODS = ['まくり差し','まくり','差し','逃げ','抜き','恵まれ']  # 長い語を先に

def normalize_text(s: str) -> str:
    return s.replace('\r','')

def normalize_name(s: str) -> str:
    # Collapse ideographic spaces and normal spaces to single space, strip ends
    s = s.replace('\t',' ')
    s = re.sub(r'[ 　]+', ' ', s)
    return s.strip()

def parse_event_header(block: str) -> Dict[str, str]:
    """
    Extract venue name, event name, day number, day label, date (YYYY-MM-DD)
    """
    # Venue + first header line
    # e.g. "大　村［成績］      1/ 1      年末年始感謝競走　　  第 4日"
    m1 = re.search(r'(?P<venue>.+?)［成績］\s+\d+/\s*\d+\s+(?P<ename>.+?)第\s*(?P<day>\d+)日', block)
    # Robustness: event name also appears on a dedicated centered line below; prefer the shorter, trimmed variant
    if m1:
        venue = normalize_name(m1.group('venue'))
        event_name = normalize_name(m1.group('ename'))
        day_num = int(m1.group('day'))
    else:
        # Fallback: venue from "ボートレース大　村"
        v2 = re.search(r'ボートレース(?P<v>.+)\s*', block)
        venue = normalize_name(v2.group('v')) if v2 else '不明場'
        # Event name from the centered banner lines (preceding/after "＊＊＊　競走成績　＊＊＊")
        e2 = re.search(r'競走成績[^\n]*\n\s*(?P<ename>.+?)\n', block)
        event_name = normalize_name(e2.group('ename')) if e2 else '不明イベント'
        day_num = None

    # Date line: "第 4日          2023/ 1/ 1                             ボートレース…"
    mdate = re.search(r'\b(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})\b', block)
    event_date = None
    if mdate:
        y, m, d = map(int, mdate.groups())
        event_date = f"{y:04d}-{m:02d}-{d:02d}"
    day_label = f"第 {day_num}日" if day_num else None

    # Clean up event_name: remove trailing "第 4日" if caught
    event_name = re.sub(r'第\s*\d+\s*日.*$', '', event_name).strip(' 　')

    return {
        'venue': venue,
        'event_name': event_name,
        'event_day_number': day_num,
        'event_day_label': day_label,
        'event_date': event_date
    }

def open_db(schema_path: Optional[Path]) -> psycopg2.extensions.connection:
    """Open a PostgreSQL connection and (optionally) apply schema.
    The schema file must be PostgreSQL-compatible. SQLite-specific lines like PRAGMA are ignored.
    """
    import os
    DB_CONF = {
        "host":     os.getenv("PGHOST", "localhost"),
        "port":     int(os.getenv("PGPORT", 5432)),
        "dbname":   os.getenv("PGDATABASE", "boatrace"),
        "user":     os.getenv("PGUSER", "br_user"),
        "password": os.getenv("PGPASSWORD", "secret"),
    }

    con = psycopg2.connect(**DB_CONF)
    con.autocommit = False
    if schema_path and schema_path.exists():
        schema_sql = schema_path.read_text(encoding='utf-8')
        # Normalize and strip comments / SQLite bits
        # 1) Drop BOM if any
        if schema_sql.startswith('\ufeff'):
            schema_sql = schema_sql.lstrip('\ufeff')
        # 2) Remove block comments /* ... */ and line comments -- ...
        no_block = re.sub(r'/\*.*?\*/', '', schema_sql, flags=re.S)
        no_line = '\n'.join([re.sub(r'--.*$', '', ln) for ln in no_block.splitlines()])
        # 3) Remove PRAGMA lines and convert AUTOINCREMENT -> SERIAL
        lines = []
        for line in no_line.splitlines():
            if not line.strip():
                continue
            if line.strip().upper().startswith('PRAGMA'):
                continue
            line = line.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
            lines.append(line)
        norm_sql = '\n'.join(lines)
        cur = con.cursor()
        for stmt in [s.strip() for s in norm_sql.split(';') if s.strip()]:
            cleaned = re.sub(r'/\*.*?\*/', '', stmt, flags=re.S)
            cleaned = re.sub(r'--.*$', '', cleaned, flags=re.M).strip()
            if not cleaned:
                continue
            cur.execute(cleaned)
    return con

def upsert_event(con, hdr: Dict[str, str], raw_header: str) -> Tuple[int, int]:
    cur = con.cursor()
    # venue
    cur.execute("INSERT INTO venue(name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (hdr['venue'],))
    cur.execute("SELECT venue_id FROM venue WHERE name=%s", (hdr['venue'],))
    venue_id = cur.fetchone()[0]
    # event
    cur.execute(
        """
        INSERT INTO event(venue_id,event_name,event_day_label,event_day_number,event_date,raw_header)
        VALUES(%s,%s,%s,%s,%s,%s)
        ON CONFLICT (venue_id, event_name, event_date) DO NOTHING
        """,
        (venue_id, hdr['event_name'], hdr['event_day_label'], hdr['event_day_number'], hdr['event_date'], raw_header)
    )
    cur.execute("SELECT event_id FROM event WHERE venue_id=%s AND event_name=%s AND event_date=%s",
                (venue_id, hdr['event_name'], hdr['event_date']))
    event_id = cur.fetchone()[0]
    con.commit()
    return venue_id, event_id

def extract_result_blocks(txt: str) -> List[str]:
    blocks: List[str] = []
    t = txt
    starts = [m.start() for m in re.finditer(r'^[^\n]*［成績］', t, re.M)]
    if starts:
        # Locate the absolute positions of all KEND markers to minimize scanning
        kend_iter = list(re.finditer(r'\n(\d{2})KEND\b|\bKEND\b', t))
        for s in starts:
            # find first KEND after s
            end_pos = None
            for m in kend_iter:
                if m.start() > s:
                    end_pos = m.end()
                    break
            blocks.append(t[s:end_pos] if end_pos else t[s:])
    else:
        for m in re.finditer(r'\b(\d{2})KBGN\b', t):
            cid = m.group(1)
            tail = t[m.end():]
            mend = re.search(rf'\b{cid}KEND\b', tail)
            end_idx = m.end() + (mend.end() if mend else 0)
            blocks.append(t[m.start(): end_idx if end_idx else len(t)])
    return blocks

def parse_race_headers(block: str) -> List[Tuple[int, int, Dict[str,str], int]]:
    lines = block.splitlines()
    idxs = []
    for i, line in enumerate(lines):
        m = re.match(r'^\s*(\d{1,2})R\s+(.+?)\s+H(\d{3,4})m\s+(\S+)\s*　?\s*風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm', line)
        if m:
            race_no = int(m.group(1))
            tail = m.group(2)
            tail_clean = normalize_name(tail)
            parts = re.split(r'\s{2,}', tail_clean)
            category = parts[0].strip()
            notes = parts[1].strip() if len(parts) > 1 else None
            meta = {
                'category': category,
                'notes': notes,
                'distance_m': int(m.group(3)),
                'weather': m.group(4),
                'wind_dir': m.group(5),
                'wind_speed': int(m.group(6)),
                'wave_cm': int(m.group(7)),
                'winning_method': None
            }
            for j in range(i+1, min(i+4, len(lines))):
                if '着' in lines[j] and '登番' in lines[j]:
                    wm = None
                    for w in WIN_METHODS:
                        if w in lines[j]:
                            wm = w; break
                    meta['winning_method'] = wm
                    break
            idxs.append((race_no, i, meta))
    out = []
    starts = [i for _, i, _ in idxs] + [len(lines)]
    for k in range(len(idxs)):
        race_no, i, meta = idxs[k]
        start_line = i
        end_line = starts[k+1]
        out.append((race_no, start_line, meta, end_line))
    return out

def insert_race(con, event_id: int, race_no: int, meta: Dict[str,str]) -> int:
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO raw.race(event_id,race_no,category,notes,distance_m,weather,wind_direction,wind_speed_m,wave_height_cm,winning_method)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (event_id, race_no) DO NOTHING
        """,
        (event_id, race_no, meta.get('category'), meta.get('notes'),
         meta.get('distance_m'), meta.get('weather'), meta.get('wind_dir'),
         meta.get('wind_speed'), meta.get('wave_cm'), meta.get('winning_method'))
    )
    cur.execute("SELECT race_id FROM raw.race WHERE event_id=%s AND race_no=%s", (event_id, race_no))
    rid = cur.fetchone()[0]
    con.commit()
    return rid

def parse_result_row(line: str) -> Optional[Dict[str, object]]:
    tail_re = re.compile(r'\s(?P<motor>\d{1,3})\s+(?P<boat>\d{1,3})\s+(?P<tenji>\d\.\d{2})\s+(?P<entry>\d)\s+(?P<start>[FS]?\d?\.\d{2})\s+(?P<rtime>(?:\d\.\d{2}\.\d)|(?:\.\s+\.\s+))\s*$')
    mtail = tail_re.search(line)
    if not mtail:
        return None
    motor = int(mtail.group('motor'))
    boat = int(mtail.group('boat'))
    tenji = float(mtail.group('tenji'))
    entry = int(mtail.group('entry'))
    start_raw = mtail.group('start')
    rtime = mtail.group('rtime').strip()
    race_time = None if rtime.startswith('.') else rtime

    pre = line[:mtail.start()].rstrip()
    mpre = re.match(r'^\s*(?P<tag>(?:\d{2}|F|S\d))\s+(?P<lane>\d)\s+(?P<reg>\d{4})\s+(?P<name>.+)$', pre)
    if not mpre:
        return None
    tag = mpre.group('tag')
    status = None
    finish_order = None
    if tag.isdigit():
        finish_order = int(tag)
    else:
        status = tag

    st_status = None
    st_time = None
    if start_raw.startswith('F'):
        st_status = 'F'
        st_time = float(start_raw[1:])
    else:
        st_time = float(start_raw)

    name = normalize_name(mpre.group('name'))

    return {
        'finish_order': finish_order,
        'status': status if status else st_status,
        'lane': int(mpre.group('lane')),
        'reg_no': int(mpre.group('reg')),
        'player_name': name,
        'motor_no': motor,
        'boat_no': boat,
        'tenji_time': tenji,
        'course_entry': entry,
        'start_timing': st_time,
        'race_time': race_time
    }

def insert_result(con, race_id: int, row: Dict[str, object]):
    cur = con.cursor()
    cur.execute("""INSERT INTO raw.result(race_id,finish_order,status,lane,reg_no,player_name,motor_no,boat_no,tenji_time,course_entry,start_timing,race_time,notes)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (race_id, row.get('finish_order'), row.get('status'), row.get('lane'),
                 row.get('reg_no'), row.get('player_name'), row.get('motor_no'),
                 row.get('boat_no'), row.get('tenji_time'), row.get('course_entry'),
                 row.get('start_timing'), row.get('race_time'), None))
    con.commit()

def parse_payout_lines(lines: List[str], start_idx: int) -> Tuple[int, List[Dict[str, object]]]:
    payouts = []
    i = start_idx
    cur_type = None

    # 券種ラベル（行頭）の検出
    label_re = re.compile(r'^(?P<label>単勝|複勝|拡連複|ワイド|二連単|２連単|2連単|二連複|２連複|2連複|三連単|３連単|3連単|三連複|３連複|3連複)\s*')
    # 例: 1-2-3 12,340円 人気 1 / 1－2 1,230 人気2 / 1 230 など
    pair_re = re.compile(
        r'(?P<combo>[1-6](?:[\-－][1-6]){0,2})\s+'
        r'(?P<payout>[\d,]+)\s*(?:円)?'
        r'(?:\s*(?:人気\s*(?P<rank>\d+)|(?P<rank2>\d+)\s*番人気))?'
    )

    while i < len(lines):
        s = lines[i]
        # 次レースやブロック終端で終了
        if re.match(r'^\s*$', s) or re.match(r'^\s*\d{1,2}R\b', s) or 'KEND' in s:
            break

        # 正規化（タブ→空白、連続空白の圧縮）
        st = s.replace('\t', ' ')
        st = re.sub(r'[ 　]+', ' ', st).strip()

        # 行頭に券種ラベルがあれば更新
        mlabel = label_re.match(st)
        if mlabel:
            cur_type = mlabel.group('label')
            st = st[mlabel.end():].strip()

        # 1行に複数ペアある場合も拾う
        for m in pair_re.finditer(st):
            combo = m.group('combo').replace('－', '-')
            payout_yen = int(m.group('payout').replace(',', ''))
            rank_txt = m.group('rank') or m.group('rank2')
            popularity_rank = int(rank_txt) if rank_txt else None
            payouts.append({
                'bet_type': cur_type,
                'combination': combo,
                'payout_yen': payout_yen,
                'popularity_rank': popularity_rank
            })
        i += 1

    return i, payouts

def load_results(con, block: str):
    hdr = parse_event_header(block)
    _, event_id = upsert_event(con, hdr, raw_header=block[:5000])

    lines = block.splitlines()
    races = parse_race_headers(block)

    for race_no, start_line, meta, end_line in races:
        race_id = insert_race(con, event_id, race_no, meta)

        # Find dashed separator line after the "着 艇 登番..." header
        i = start_line + 1
        while i < end_line and '----' not in lines[i]:
            i += 1
        if i >= end_line:
            continue
        i += 1  # move to first result row

        # Parse result rows until we hit empty line or payout header
        while i < end_line:
            line = lines[i]
            if any(line.strip().startswith(t) for t in BET_TYPES) or line.strip() == '' or re.match(r'^\s*\d{1,2}R\b', line):
                break
            row = parse_result_row(line)
            if row:
                insert_result(con, race_id, row)
            i += 1

        # 空行をスキップして最初の券種行へ
        while i < end_line and lines[i].strip() == '':
            i += 1

        # Parse payouts (if present)
        if i < end_line and any(lines[i].lstrip().replace('\t',' ').lstrip(' \u3000').startswith(t) for t in BET_TYPES):
            i, payouts = parse_payout_lines(lines, i)
            if payouts:
                cur = con.cursor()
                cur.executemany("""INSERT INTO raw.payout(race_id,bet_type,combination,payout_yen,popularity_rank)
                                   VALUES(%s,%s,%s,%s,%s)""",
                                [(race_id, p['bet_type'], p['combination'], p['payout_yen'], p['popularity_rank']) for p in payouts])
                con.commit()

def main():
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    files = []
    dir_path = f'download/txt/results'
    schema_path = 'sql/02_boat_results_schema.sql'
    root = Path(dir_path)
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() == '.txt':
            files.append(p)

    # Deduplicate and sort
    files = sorted({str(p): p for p in files}.values())

    if not files:
        print('[ERROR] No input files found. Use --txt or --dir.', file=sys.stderr)
        return

    con = open_db(Path(schema_path) if schema_path else None)

    ok = 0
    skipped = 0
    try:
        for f in files:
            try:
                content = Path(f).read_text(encoding='utf-8', errors='ignore')
                content = normalize_text(content)
                blocks = extract_result_blocks(content)
                if not blocks:
                    raise RuntimeError('no results block found')
                for bi, block in enumerate(blocks, start=1):
                    load_results(con, block)
                    hdr = parse_event_header(block)
                    v = hdr.get('venue') or '?'
                    en = hdr.get('event_name') or '?'
                    ed = hdr.get('event_date') or '?'
                    print(f"[OK] {f}  block {bi}/{len(blocks)}  -> {v} {en} {ed}")
                ok += 1
            except Exception as e:
                print(f"[SKIP] {f} -> {e}")
                skipped += 1
        con.commit()
    finally:
        con.close()

    print(f"Done. Imported: {ok}, Skipped: {skipped}")

if __name__ == '__main__':
    main()