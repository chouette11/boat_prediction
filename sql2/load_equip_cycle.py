#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV -> raw.equip_cycles ローダ（最小版）
PK: (stadium, equip_type, start_date) で ON CONFLICT DO NOTHING

使い方:
  export PGHOST=localhost PGPORT=5432 PGDATABASE=boat PGUSER=postgres PGPASSWORD=secret
  python load_equip_cycles.py equip_cycles.csv
"""

import os, sys, csv
from datetime import datetime
import psycopg2

REQUIRED_HEADERS = ["stadium", "equip_type", "start_date", "next_start_date", "note"]
VALID_EQUIP_TYPES = {"motor", "boat"}
DATE_INPUT_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d")

def parse_date(s: str):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    for fmt in DATE_INPUT_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    # そのまま返してPostgresに任せる（基本は到達しない想定）
    return s

def main():
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    csv_path = "download/equip_cycles.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE"),
        user=os.environ.get("PGUSER"),
        password=os.environ.get("PGPASSWORD"),
    )
    conn.autocommit = False

    try:
        with conn, conn.cursor() as cur, open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = [h.strip().lower() for h in reader.fieldnames or []]
            missing = [h for h in REQUIRED_HEADERS if h not in headers]
            if missing:
                raise RuntimeError(f"CSVヘッダ不足: {missing} （必要: {REQUIRED_HEADERS}）")

            rows = []
            for i, r in enumerate(reader, start=2):  # 2 = ヘッダ行の次
                stadium = (r.get("stadium") or "").strip()
                equip_type = (r.get("equip_type") or "").strip().lower()
                start_date = parse_date(r.get("start_date"))
                next_start_date = parse_date(r.get("next_start_date"))
                note = (r.get("note") or "").strip()

                if not stadium or not start_date or equip_type not in VALID_EQUIP_TYPES:
                    raise RuntimeError(
                        f"{i}行目が不正: stadium={stadium!r}, equip_type={equip_type!r}, start_date={start_date!r}"
                    )
                rows.append((stadium, equip_type, start_date, next_start_date, note))

            sql = """
                INSERT INTO raw.equip_cycles
                    (stadium, equip_type, start_date, next_start_date, note)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (stadium, equip_type, start_date) DO NOTHING
            """
            cur.executemany(sql, rows)
        print(f"✅ inserted {len(rows)} rows into raw.equip_cycles")
    except Exception as e:
        conn.rollback()
        print(f"❌ error: {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        conn.close()

if __name__ == "__main__":
    main()