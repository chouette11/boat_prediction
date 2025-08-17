# parse_programs.py
# -*- coding: utf-8 -*-

import argparse
import psycopg2
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import sys


def nfkc(s: str) -> str:
    return unicodedata.normalize('NFKC', s)


def normalize_spaces(s: str) -> str:
    return re.sub(r'[ \t\u3000]+', ' ', s).strip()


def normalize_text(s: str) -> str:
    return s.replace('\r', '')


def extract_program_blocks(txt: str) -> List[str]:
    """Extract ALL program blocks from a B*.TXT file using NNVVBBGN..NNVV BEND markers.
    Primary: pairs of 'NNBBGN'...'NNBEND' (e.g., '24BBGN'..'24BEND').
    Fallback: scan for the banner '＊＊＊ 番組表 ＊＊＊' and cut until next 'NNBEND' or end of file.
    """
    blocks: List[str] = []
    t = txt
    for m in re.finditer(r"\b(\d{2})BBGN\b", t):
        cid = m.group(1)
        tail = t[m.end():]
        mend = re.search(rf"\b{cid}BEND\b", tail)
        end_idx = m.end() + (mend.end() if mend else 0)
        blocks.append(t[m.start(): end_idx if end_idx else len(t)])
    if blocks:
        return blocks
    # Fallback: banner-based detection
    starts = [m.start() for m in re.finditer(r"＊＊＊\s*番組表\s*＊＊＊", t)]
    if starts:
        kend_iter = list(re.finditer(r"\n(\d{2})BEND\b|\bBEND\b", t))
        for s in starts:
            end_pos = None
            for m in kend_iter:
                if m.start() > s:
                    end_pos = m.end(); break
            blocks.append(t[s:end_pos] if end_pos else t[s:])
    return blocks


def parse_event_header(block: str) -> Dict[str, Optional[str]]:
    """Extract venue, event_name, day number/label, date (YYYY-MM-DD) from a program block."""
    b = nfkc(block)
    mvenue = re.search(r'ボートレース(?P<v>.+?)\s', b)
    venue = normalize_spaces(mvenue.group('v')) if mvenue else '不明場'

    m_banner = re.search(r'＊＊＊\s*番組表\s*＊＊＊', b)
    event_name = None
    if m_banner:
        after = b[m_banner.end():].splitlines()
        for line in after:
            line = normalize_spaces(line)
            if line:
                event_name = line; break

    mday = re.search(r'第\s*(\d+)\s*日', b)
    day_num = int(mday.group(1)) if mday else None
    day_label = f"第 {day_num}日" if day_num else None

    mdate = re.search(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', b)
    if not mdate:
        mmd = re.search(r'\b(\d{1,2})月\s*(\d{1,2})日\b', b)
        y = None
        if mmd:
            # Heuristic: look for YYYY/.. elsewhere
            y1 = re.search(r'\b(\d{4})年\b', b)
            y = int(y1.group(1)) if y1 else None
            if not y:
                # last resort: try to sniff from file naming convention outside
                y = 2000
            m, d = map(int, mmd.groups())
        else:
            y, m, d = (2000, 1, 1)
    else:
        y, m, d = map(int, mdate.groups())
    event_date = f"{y:04d}-{m:02d}-{d:02d}"

    return {
        'venue': venue,
        'event_name': event_name or '不明イベント',
        'event_day_number': day_num,
        'event_day_label': day_label,
        'event_date': event_date
    }


def open_db(schema_path: Optional[Path]):
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
        norm = '\n'.join(lines)
        # 4) Split into statements and skip empties again just in case
        stmts = [s.strip() for s in norm.split(';') if s.strip()]
        cur = con.cursor()
        for stmt in stmts:
            # if a chunk still contains only comments/whitespace, skip
            cleaned = re.sub(r'/\*.*?\*/', '', stmt, flags=re.S)
            cleaned = re.sub(r'--.*$', '', cleaned, flags=re.M).strip()
            if not cleaned:
                continue
            cur.execute(cleaned)
        con.commit()
    return con


def upsert_event(con, hdr: Dict[str, Optional[str]], raw_header: str) -> Tuple[int, int]:
    cur = con.cursor()
    cur.execute("INSERT INTO venue(name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (hdr['venue'],))
    cur.execute("SELECT venue_id FROM venue WHERE name=%s", (hdr['venue'],))
    venue_id = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO event(venue_id,event_name,event_day_label,event_day_number,event_date,raw_header)
        VALUES(%s,%s,%s,%s,%s,%s)
        ON CONFLICT (venue_id, event_name, event_date) DO NOTHING
        """,
        (venue_id, hdr['event_name'], hdr['event_day_label'], hdr['event_day_number'], hdr['event_date'], raw_header[:8000])
    )
    cur.execute("SELECT event_id FROM event WHERE venue_id=%s AND event_name=%s AND event_date=%s",
                (venue_id, hdr['event_name'], hdr['event_date']))
    event_id = cur.fetchone()[0]
    con.commit()
    return venue_id, event_id


race_header_re = re.compile(
    r'^\s*(?P<race_no>\d{1,2})R\s+(?P<category>\S+)\s*(?P<fix>進入固定)?\s*H(?P<dist>\d{3,4})m\s+.*?(\d{1,2}):(\d{2})',
    re.IGNORECASE
)


def find_race_spans(block: str) -> List[Tuple[int, int, int]]:
    """Return list of (race_no, start_line_index, end_line_index) for each race block."""
    lines = block.splitlines()
    idxs = []
    for i, line in enumerate(lines):
        if race_header_re.search(nfkc(line)):
            idxs.append(i)
    idxs.append(len(lines))
    spans = []
    for j in range(len(idxs)-1):
        start = idxs[j]
        end = idxs[j+1]
        while end > start and not lines[end-1].strip():
            end -= 1
        m = race_header_re.search(nfkc(lines[start]))
        if m:
            spans.append((int(m.group('race_no')), start, end))
    return spans


def insert_program_race(con, event_id: int, header_line: str, raw_block: str) -> int:
    s = nfkc(header_line)
    m = race_header_re.search(s)
    race_no = int(m.group('race_no'))
    category = m.group('category')
    entry_fixed = 1 if m.group('fix') else 0
    distance_m = int(m.group('dist'))
    hm = re.search(r'(\d{1,2}):(\d{2})', s)
    phone_close_time = f"{int(hm.group(1)):02d}:{int(hm.group(2)):02d}" if hm else None

    cur = con.cursor()
    cur.execute("SELECT event_date FROM event WHERE event_id=%s", (event_id,))
    event_date = cur.fetchone()[0]
    phone_close_at = f"{event_date} {phone_close_time}:00" if phone_close_time else None

    cur.execute(
        """
        INSERT INTO raw.program_race(event_id,race_no,category,notes,entry_fixed,distance_m,phone_close_time,phone_close_at,raw_block)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (event_id, race_no) DO NOTHING
        """,
        (event_id, race_no, category, None, entry_fixed, distance_m, phone_close_time, phone_close_at, raw_block[:6000])
    )
    cur.execute("SELECT program_race_id FROM raw.program_race WHERE event_id=%s AND race_no=%s", (event_id, race_no))
    prid = cur.fetchone()[0]
    con.commit()
    return prid


entry_line_core_re = re.compile(
    r'^\s*(?P<lane>\d)\s+(?P<reg>\d{4})(?P<name>.+?)(?P<age>\d{1,2})(?P<branch>[^\d\s]+?)(?P<weight>\d{2})(?P<class>[AB]\d)\s+(?P<rest>.+)$'
)


def parse_entry_line(line: str) -> Optional[Dict[str, object]]:
    s = nfkc(line).rstrip()
    m = entry_line_core_re.match(s)
    if not m:
        return None

    lane = int(m.group('lane'))
    reg_no = int(m.group('reg'))
    name = normalize_spaces(m.group('name'))
    age = int(m.group('age'))
    branch = m.group('branch')
    weight = int(m.group('weight'))
    clazz = m.group('class')
    rest = m.group('rest').strip()

    toks = re.findall(r'[FS]?\d+(?:\.\d+)?|[FS]+|[A-Za-z]+', rest)

    def pop_float() -> Optional[float]:
        if not toks:
            return None
        return float(toks.pop(0))

    def pop_int() -> Optional[int]:
        if not toks:
            return None
        t = toks.pop(0)
        return int(re.sub(r'[^0-9]', '', t)) if re.search(r'\d', t) else None

    nat_win = pop_float()
    nat_2 = pop_float()
    local_win = pop_float()
    local_2 = pop_float()
    motor_no = pop_int()
    motor_2 = pop_float()
    boat_no = pop_int()
    boat_2 = pop_float()

    series_note = None
    early_note = None
    if toks:
        if len(toks) >= 2 and re.fullmatch(r'\d{1,2}', toks[-1]):
            early_note = toks[-1]
            series_note = ' '.join(toks[:-1])
        else:
            series_note = ' '.join(toks)

    return {
        'lane': lane,
        'reg_no': reg_no,
        'player_name': name,
        'age': age,
        'branch': branch,
        'weight_kg': weight,
        'class': clazz,
        'nat_win_rate': nat_win,
        'nat_2rate': nat_2,
        'local_win_rate': local_win,
        'local_2rate': local_2,
        'motor_no': motor_no,
        'motor_2rate': motor_2,
        'boat_no': boat_no,
        'boat_2rate': boat_2,
        'series_note': series_note,
        'early_note': early_note,
    }


def load_program(con, block: str):
    hdr = parse_event_header(block)
    _, event_id = upsert_event(con, hdr, raw_header=block[:8000])

    lines = block.splitlines()
    spans = find_race_spans(block)

    for race_no, i, j in spans:
        header_line = lines[i]
        race_chunk = '\n'.join(lines[i:j])
        prid = insert_program_race(con, event_id, header_line, race_chunk)

        k = i + 1
        while k < j and not re.match(r'-{5,}', nfkc(lines[k])):
            k += 1
        k += 1
        while k < j and not re.match(r'-{5,}', nfkc(lines[k])):
            k += 1
        k += 1  # first entry row

        row_count = 0
        while k < j and row_count < 6:
            s = lines[k]
            if not s.strip():
                break
            if race_header_re.search(nfkc(s)):
                break
            ent = parse_entry_line(s)
            if ent:
                cur = con.cursor()
                cur.execute(
                    """
                    INSERT INTO raw.program_entry(
                        program_race_id,lane,reg_no,player_name,age,branch,weight_kg,class,
                        nat_win_rate,nat_2rate,local_win_rate,local_2rate,
                        motor_no,motor_2rate,boat_no,boat_2rate,series_note,early_note
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (program_race_id, lane) DO UPDATE SET
                        reg_no=EXCLUDED.reg_no,
                        player_name=EXCLUDED.player_name,
                        age=EXCLUDED.age,
                        branch=EXCLUDED.branch,
                        weight_kg=EXCLUDED.weight_kg,
                        class=EXCLUDED.class,
                        nat_win_rate=EXCLUDED.nat_win_rate,
                        nat_2rate=EXCLUDED.nat_2rate,
                        local_win_rate=EXCLUDED.local_win_rate,
                        local_2rate=EXCLUDED.local_2rate,
                        motor_no=EXCLUDED.motor_no,
                        motor_2rate=EXCLUDED.motor_2rate,
                        boat_no=EXCLUDED.boat_no,
                        boat_2rate=EXCLUDED.boat_2rate,
                        series_note=EXCLUDED.series_note,
                        early_note=EXCLUDED.early_note
                    """,
                    (
                        prid, ent['lane'], ent['reg_no'], ent['player_name'], ent['age'], ent['branch'], ent['weight_kg'], ent['class'],
                        ent['nat_win_rate'], ent['nat_2rate'], ent['local_win_rate'], ent['local_2rate'],
                        ent['motor_no'], ent['motor_2rate'], ent['boat_no'], ent['boat_2rate'], ent['series_note'], ent['early_note']
                    )
                )
                con.commit()
                row_count += 1
            k += 1


def main():
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    # Collect files
    files = []
    dir_path = 'download/txt/programs/2022'
    schema_path = 'sql2/02_boat_programs_schema.sql'
    root = Path(dir_path)
    if not root.exists() or not root.is_dir():
        print(f"[WARN] --dir not found or not a directory: {root}", file=sys.stderr)
    else:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() == '.txt':
                files.append(p)


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
                blocks = extract_program_blocks(content)
                if not blocks:
                    raise RuntimeError('no program block found')
                for bi, block in enumerate(blocks, start=1):
                    load_program(con, block)
                    hdr = parse_event_header(block)
                    v = hdr.get('venue') or '?'
                    en = hdr.get('event_name') or '?'
                    ed = hdr.get('event_date') or '?'
                    print(f"[OK] {f}  block {bi}/{len(blocks)}  -> {v} {en} {ed}")
                ok += 1
            except Exception as e:
                con.rollback()
                print(f"[SKIP] {f} -> {e}")
                skipped += 1
        con.commit()
    finally:
        con.close()

    print(f"Done. Imported: {ok}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
