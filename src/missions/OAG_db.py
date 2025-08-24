import logging
import os
import sqlite3
from collections.abc import Generator
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd

import utils.airports as airports
from missions.custom_types import DayOfWeek
from parsers.OAG_reader import OAGEntry
from utils.units import STATUTE_MILES_TO_KM

logger = logging.getLogger(__name__)


class Comparison(str, Enum):
    EQUAL = '=='
    NOT_EQUAL = '!='
    GREATER_THAN = '>'
    LESS_THAN = '<'
    GREATER_OR_EQUAL = '>='
    LESS_OR_EQUAL = '<='

    @property
    def sql(self):
        if self == Comparison.EQUAL:
            return '='
        elif self == Comparison.NOT_EQUAL:
            return '<>'
        else:
            return self.value


@dataclass
class Condition:
    field: str
    value: str
    comp: Comparison

    def __str__(self):
        return f'{self.field}{self.comp.value}{self.value}'


class OAGDatabase:
    def __init__(self, db_path: str, write_mode: bool = False):
        if os.path.exists(db_path):
            if write_mode:
                raise RuntimeError(f'Database file {db_path} already exists.')
        else:
            if not write_mode:
                raise RuntimeError(f'Database file {db_path} does not exist.')
        self.db_path = db_path
        self.write_mode = write_mode
        self.conn = sqlite3.connect(self.db_path)

        # Foreign key constraints are enabled at the connection level, so this
        # needs to be done every time we connect to the database.
        self.cursor().execute("PRAGMA foreign_keys = ON")

        if self.write_mode:
            self._ensure_schema(indexes=False)

    def index(self):
        if self.write_mode:
            self._ensure_schema(indexes=True)

    def cursor(self):
        assert self.conn is not None, 'DB connection is not open'
        return self.conn.cursor()

    def add(self, e: OAGEntry, commit: bool = True):
        if not self.write_mode:
            raise RuntimeError("Database is not in write mode")
        cur = self.conn.cursor()

        # Add airport entries if they don't already exist. (Also adds country
        # entries as needed.)
        origin_id = self._get_or_add_airport(e.depapt, cur)
        destination_id = self._get_or_add_airport(e.arrapt, cur)
        if origin_id is None or destination_id is None:
            logger.warn(
                'Skipping flight with unknown airport: %s -> %s', e.depapt, e.arrapt
            )
            return

        # Add flight entry (number of flights to be computed from schedule).
        flight_id = self._add_flight(e, origin_id, destination_id, cur)

        # Add schedule entries.
        num_flights = self._add_schedule(e, flight_id, cur)

        # Set number of flights in flight entry.
        cur.execute(
            'UPDATE flights SET number_of_flights = ? WHERE id = ?',
            (num_flights, flight_id),
        )

        if commit:
            self.conn.commit()

    def _add_schedule(self, e: OAGEntry, flight_id: int, cur: sqlite3.Cursor) -> int:
        num_flights = 0
        for d in pd.date_range(e.efffrom, e.effto):
            if DayOfWeek.from_pandas(d) not in e.days:
                continue
            departure_timestamp = int(
                (
                    d + timedelta(hours=e.deptim.hour, minutes=e.deptim.minute)
                ).timestamp()
            )
            arrival_timestamp = int(
                (
                    d + timedelta(hours=e.arrtim.hour, minutes=e.arrtim.minute)
                ).timestamp()
            )
            if arrival_timestamp < departure_timestamp:
                # Arrival is on the next day.
                arrival_timestamp += 24 * 3600
            day = (d - datetime(1970, 1, 1)).days

            cur.execute(
                """INSERT INTO schedules (
                    departure_timestamp, arrival_timestamp, day, flight_id
                ) VALUES (?, ?, ?, ?)""",
                (departure_timestamp, arrival_timestamp, day, flight_id),
            )
            num_flights += 1
        return num_flights

    def _add_flight(
        self, e: OAGEntry, origin_id: int, destination_id: int, cur: sqlite3.Cursor
    ) -> int:
        placeholders = '?, ' * 13 + '?'
        cur.execute(
            f"""
            INSERT INTO flights (
                carrier, flight_number, origin, destination,
                day_of_week_mask, departure_time, arrival_time,
                aircraft_type, engine_type, distance, seat_capacity,
                effective_from, effective_to, number_of_flights
            ) VALUES ({placeholders})
            RETURNING id""",
            (
                e.carrier,
                e.fltno,
                origin_id,
                destination_id,
                self._make_dow_mask(e.days),
                e.deptim.hour * 60 + e.deptim.minute,
                e.arrtim.hour * 60 + e.arrtim.minute,
                e.inpacft,
                '',  # Engine type not available.
                e.distance * STATUTE_MILES_TO_KM,
                e.seats,
                e.efffrom.isoformat(),
                e.effto.isoformat(),
                0,  # Number of flights to be computed from schedule.
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return row[0]

    def _get_or_add_airport(self, iata_code: str, cur: sqlite3.Cursor) -> int | None:
        cur.execute('SELECT id FROM airports WHERE iata_code = ?', (iata_code,))
        row = cur.fetchone()
        if row:
            return row[0]

        airport = airports.airports[iata_code]
        if not airport:
            logger.warn('Unknown airport IATA code: %s', iata_code)
            return None

        country = self._get_or_add_country(airport.country, cur)

        cur.execute(
            """INSERT INTO airports (
                iata_code, name, municipality, country,
                latitude, longitude, elevation
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id""",
            (
                airport.iata_code,
                airport.name,
                airport.municipality,
                country,
                airport.latitude,
                airport.longitude,
                airport.elevation,
            ),
        )
        row = cur.fetchone()
        assert row is not None
        airport_id = row[0]

        # Add to R-Tree index.
        cur.execute(
            """INSERT INTO airport_location_idx (
                id, min_latitude, max_latitude,
                min_longitude, max_longitude
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                airport_id,
                airport.latitude,
                airport.latitude,
                airport.longitude,
                airport.longitude,
            ),
        )

        return airport_id

    def _get_or_add_country(self, code: str, cur: sqlite3.Cursor) -> str:
        cur.execute('SELECT code FROM countries WHERE code = ?', (code,))
        row = cur.fetchone()
        if row:
            return row[0]

        country = airports.countries[code]
        if not country:
            raise ValueError(f'Unknown country code: {code}')

        cur.execute(
            """INSERT INTO countries (code, name, continent)
               VALUES (?, ?, ?)""",
            (country.code, country.name, country.continent),
        )
        return country.code

    @staticmethod
    def _make_dow_mask(days: set[DayOfWeek]) -> int:
        return sum(1 << (day.value - 1) for day in days)

    def __call__(self, *conds: Condition) -> Generator[OAGEntry, None, None]:
        conditions = []
        parameters = []
        valid_fields = set(f.name for f in fields(OAGEntry))
        for c in conds:
            if c.field not in valid_fields:
                raise ValueError(f'Invalid field: {c.field}')
            conditions.append(f'{c.field} {c.comp.sql} ?')
            parameters.append(c.value)

        sql = 'SELECT * FROM entries'
        if len(conditions) > 0:
            sql += ' WHERE ' + ' AND '.join(conditions)

        cur = self.conn.cursor()
        for row in cur.execute(sql, parameters):
            yield OAGEntry.from_db_row(row)

    def _ensure_schema(self, indexes: bool = True):
        cur = self.conn.cursor()

        logger.info('Creating OAG database tables')

        cur.execute("""
          CREATE TABLE IF NOT EXISTS flights (
            id INTEGER PRIMARY KEY NOT NULL,
            carrier TEXT NOT NULL,
            flight_number TEXT NOT NULL,
            origin INTEGER NOT NULL REFERENCES airports(id),
            destination INTEGER NOT NULL REFERENCES airports(id),
            day_of_week_mask INT NOT NULL,
            departure_time INTEGER NOT NULL,
            arrival_time INTEGER NOT NULL,
            aircraft_type TEXT NOT NULL,
            engine_type TEXT,
            distance REAL NOT NULL,
            seat_capacity INT NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to TEXT NOT NULL,
            number_of_flights INT NOT NULL
          )""")

        cur.execute("""
          CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY NOT NULL,
            departure_timestamp INTEGER NOT NULL,
            arrival_timestamp INTEGER NOT NULL,
            day INTEGER NOT NULL,
            flight_id INTEGER NOT NULL REFERENCES flights(id)
          )""")

        cur.execute("""
          CREATE TABLE IF NOT EXISTS airports (
            id INTEGER PRIMARY KEY NOT NULL,
            iata_code TEXT NOT NULL,
            name TEXT NOT NULL,
            municipality TEXT,
            country TEXT NOT NULL REFERENCES countries(code),
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            elevation REAL
          )""")

        cur.execute("""
          CREATE VIRTUAL TABLE IF NOT EXISTS airport_location_idx USING rtree(
            id INTEGER PRIMARY KEY,
            min_latitude, max_latitude,
            min_longitude, max_longitude
          )""")

        cur.execute("""
          CREATE TABLE IF NOT EXISTS countries (
            code TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            continent TEXT NOT NULL
          )""")

        if not indexes:
            return

        logger.info('Creating OAG database indexes')

        index_data = [
            ('flights', 'carrier', False),
            ('flights', 'origin', False),
            ('flights', 'destination', False),
            ('flights', 'aircraft_type', False),
            ('flights', 'distance', False),
            ('flights', 'seat_capacity', False),
            ('schedules', 'departure_timestamp', False),
            ('schedules', 'flight_id', False),
            ('airports', 'iata_code', True),
            ('airports', 'country', False),
        ]
        for table, column, unique in index_data:
            cur.execute(f"""
              CREATE {'UNIQUE' if unique else ''} INDEX IF NOT EXISTS
                  {table}_{column}_idx ON {table}({column})""")
