"""Rebuild data/soretrak_gtfs.db from the raw GTFS files.
"""
import csv
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DB_PATH = ROOT_DIR / "data" / "soretrak_gtfs.db"
DATA_DIR = ROOT_DIR / "data" / "gtfs"
SCHEMA_PATH = ROOT_DIR / "data" / "sql" / "schema.sql"
VIEWS_PATH = ROOT_DIR / "data" / "sql" / "views.sql"

# Import order: each table only depends on the tables before it.
IMPORT_ORDER = ["agency", "calendar", "routes", "stops", "shapes", "trips", "stop_times"]

# Columns inserted per table, in schema order ("shapes" has an auto-increment
# id that GTFS does not provide, so it is not inserted).
TABLE_COLUMNS = {
    "agency": ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_phone"],
    "calendar": [
        "service_id", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "start_date", "end_date",
    ],
    "routes": ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
    "stops": ["stop_id", "stop_name", "stop_lat", "stop_lon"],
    "shapes": ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
    "trips": ["trip_id", "route_id", "service_id", "trip_headsign", "direction_id", "shape_id"],
    "stop_times": [
        "trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
        "stop_headsign", "pickup_type", "shape_dist_traveled", "timepoint",
    ],
}


def read_rows(table_name: str) -> list[dict]:
    """Read a GTFS .txt file and return its rows, with stripped values."""
    file_path = DATA_DIR / f"{table_name}.txt"
    if not file_path.exists():
        print(f"[ERREUR] fichier introuvable : {file_path}")
        return []
    with open(file_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def import_gtfs() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    valid_agency_ids: set[str] = set()
    valid_service_ids: set[str] = set()
    valid_route_ids: set[str] = set()
    valid_stop_ids: set[str] = set()
    valid_trip_ids: set[str] = set()

    for table_name in IMPORT_ORDER:
        rows = read_rows(table_name)
        original_count = len(rows)

        if table_name == "agency":
            valid_agency_ids = {r["agency_id"] for r in rows}

        elif table_name == "calendar":
            valid_service_ids = {r["service_id"] for r in rows}

        elif table_name == "routes":
            rows = [r for r in rows if r.get("agency_id", "") in valid_agency_ids]
            valid_route_ids = {r["route_id"] for r in rows}

        elif table_name == "stops":
            valid_stop_ids = {r["stop_id"] for r in rows}

        elif table_name == "trips":
            rows = [
                r for r in rows
                if r["route_id"] in valid_route_ids and r["service_id"] in valid_service_ids
            ]
            valid_trip_ids = {r["trip_id"] for r in rows}

        elif table_name == "stop_times":
            rows = [
                r for r in rows
                if r["trip_id"] in valid_trip_ids and r["stop_id"] in valid_stop_ids
            ]

        columns = TABLE_COLUMNS[table_name]
        placeholders = ", ".join("?" for _ in columns)
        values = [tuple(r.get(c, "") or None for c in columns) for r in rows]

        if values:
            conn.executemany(
                f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )

        filtered_count = original_count - len(rows)
        print(f"[OK] {table_name:<12} {len(rows):>8} lignes importées (filtrées : {filtered_count})")

    conn.commit()
    conn.close()


def build() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.close()
    print("[OK] schéma créé")

    import_gtfs()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(VIEWS_PATH.read_text(encoding="utf-8"))
    conn.close()
    print("[OK] vues créées")

    print(f"\nBase reconstruite : {DB_PATH}")


if __name__ == "__main__":
    build()
