-- boat_results_schema.sql (PostgreSQL)

-- NOTE: Run this on PostgreSQL. SQLite-specific directives have been removed.

CREATE SCHEMA IF NOT EXISTS raw;

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

CREATE TABLE IF NOT EXISTS raw.race (
  race_id SERIAL PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES event(event_id),
  race_no INTEGER NOT NULL,
  category TEXT,
  notes TEXT,
  distance_m INTEGER,
  weather TEXT,
  wind_direction TEXT,
  wind_speed_m INTEGER,
  wave_height_cm INTEGER,
  winning_method TEXT,
  UNIQUE(event_id, race_no)
);

CREATE TABLE IF NOT EXISTS raw.result (
  result_id SERIAL PRIMARY KEY,
  race_id INTEGER NOT NULL REFERENCES raw.race(race_id),
  finish_order INTEGER,
  status TEXT,
  lane INTEGER,
  reg_no INTEGER,
  player_name TEXT,
  motor_no INTEGER,
  boat_no INTEGER,
  tenji_time REAL,
  course_entry INTEGER,
  start_timing REAL,
  race_time TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS raw.payout (
  payout_id SERIAL PRIMARY KEY,
  race_id INTEGER NOT NULL REFERENCES raw.race(race_id),
  bet_type TEXT NOT NULL,
  combination TEXT NOT NULL,
  payout_yen INTEGER,
  popularity_rank INTEGER
);

CREATE INDEX IF NOT EXISTS idx_event_venue_date ON event(venue_id, event_date);
CREATE INDEX IF NOT EXISTS idx_race_event ON raw.race(event_id);
CREATE INDEX IF NOT EXISTS idx_result_race ON raw.result(race_id);
CREATE INDEX IF NOT EXISTS idx_payout_race ON raw.payout(race_id);