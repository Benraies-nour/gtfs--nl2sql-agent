"""The database rebuilds, the views exist, the examples run."""
import sqlite3
from pathlib import Path

import pytest
import yaml

ROOT_DIR = Path(__file__).parent.parent
DB_PATH = ROOT_DIR / "data" / "soretrak_gtfs.db"
EXAMPLES_PATH = ROOT_DIR / "soretrak" / "knowledge" / "examples.yaml"


@pytest.fixture(scope="module")
def conn():
    """The database must already exist (built by scripts/build_db.py).

    It is not rebuilt here: build_db.py is tested separately for
    reproducibility, not on every run of the suite.
    """
    if not DB_PATH.exists():
        pytest.skip("data/soretrak_gtfs.db absente — lancer scripts/build_db.py d'abord")
    c = sqlite3.connect(DB_PATH)
    yield c
    c.close()


def test_views_exist(conn):
    views = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
    }
    assert "v_departures" in views
    assert "v_route_stops" in views
    assert "v_trajets" in views


def test_v_departures_not_empty(conn):
    assert conn.execute("SELECT COUNT(*) FROM v_departures").fetchone()[0] > 0


def test_v_route_stops_not_empty(conn):
    assert conn.execute("SELECT COUNT(*) FROM v_route_stops").fetchone()[0] > 0


def test_v_trajets_keeps_order_and_real_durations(conn):
    """A always precedes B, so every duration is positive (no text-time arithmetic)."""
    assert conn.execute("SELECT MIN(duree_min) FROM v_trajets").fetchone()[0] > 0
    rows = conn.execute(
        "SELECT route_id, MIN(duree_min) FROM v_trajets WHERE depart_id = '429' "
        "AND arrivee_id = '432' GROUP BY route_id ORDER BY route_id").fetchall()
    assert rows == [("21", 16), ("22", 32), ("70", 20)]
    # Bab Khoukha → Bab Jedid: line 21 runs the other way and must not appear.
    lines = {r[0] for r in conn.execute(
        "SELECT route_id FROM v_trajets WHERE depart_id = '433' AND arrivee_id = '432'")}
    assert lines == {"3", "22", "61", "68"}


def test_no_orphan_rows(conn):
    """The import filters let nothing invalid through."""
    orphan_trips = conn.execute(
        "SELECT COUNT(*) FROM trips WHERE route_id NOT IN (SELECT route_id FROM routes)"
    ).fetchone()[0]
    assert orphan_trips == 0

    orphan_stop_times = conn.execute(
        "SELECT COUNT(*) FROM stop_times WHERE trip_id NOT IN (SELECT trip_id FROM trips)"
    ).fetchone()[0]
    assert orphan_stop_times == 0


@pytest.fixture(scope="module")
def examples():
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_examples_file_has_entries(examples):
    assert len(examples) >= 6


def test_each_example_returns_rows(conn, examples):
    for example in examples:
        rows = conn.execute(example["sql"]).fetchall()
        assert len(rows) > 0, f"exemple sans résultat : {example['question']!r}"
