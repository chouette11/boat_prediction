
-- boat_programs_schema.sql (PostgreSQL)

-- Reuse venue & event if they already exist from results schema; otherwise create.
CREATE TABLE IF NOT EXISTS venue (
  venue_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS event (
  event_id SERIAL PRIMARY KEY,
  venue_id INTEGER NOT NULL REFERENCES venue(venue_id),
  event_name TEXT NOT NULL,
  event_day_label TEXT,
  event_day_number INTEGER,
  event_date DATE NOT NULL,
  raw_header TEXT,
  UNIQUE(venue_id, event_name, event_date)
);

-- Program race info (one row per R in 番組表)
DROP TABLE IF EXISTS raw.program_race CASCADE;
CREATE TABLE IF NOT EXISTS raw.program_race (
  program_race_id SERIAL PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES event(event_id),
  race_no INTEGER NOT NULL,
  category TEXT,
  notes TEXT,
  entry_fixed INTEGER DEFAULT 0,
  distance_m INTEGER,
  phone_close_time TEXT,
  phone_close_at TEXT,
  raw_block TEXT,
  UNIQUE(event_id, race_no)
);

-- Entries shown in 番組表 for each race lane (1..6)
DROP TABLE IF EXISTS raw.program_entry CASCADE;
CREATE TABLE IF NOT EXISTS raw.program_entry (
  program_entry_id SERIAL PRIMARY KEY,
  program_race_id INTEGER NOT NULL REFERENCES raw.program_race(program_race_id) ON DELETE CASCADE,
  lane INTEGER NOT NULL,
  reg_no INTEGER NOT NULL,
  player_name TEXT NOT NULL,
  age INTEGER,
  branch TEXT,
  weight_kg INTEGER,
  class TEXT,
  nat_win_rate REAL,
  nat_2rate REAL,
  local_win_rate REAL,
  local_2rate REAL,
  motor_no INTEGER,
  motor_2rate REAL,
  boat_no INTEGER,
  boat_2rate REAL,
  series_note TEXT,
  early_note TEXT,
  UNIQUE(program_race_id, lane)
);

CREATE INDEX IF NOT EXISTS idx_program_race_event ON raw.program_race(event_id);
CREATE INDEX IF NOT EXISTS idx_program_entry_race ON raw.program_entry(program_race_id);
