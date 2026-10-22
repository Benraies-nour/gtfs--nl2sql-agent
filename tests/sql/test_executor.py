"""run_query against a real database (throwaway copy): no write may get
through, and all the dangerous cases from the audit are blocked."""
import shutil
import sqlite3
import time

import pytest

from soretrak.config import ROOT

from soretrak.sql.executor import run_query

ORIGINAL_DB = ROOT / "data" / "soretrak_gtfs.db"


@pytest.fixture
def db_copy(tmp_path):
    """A throwaway copy of the real database, a different folder per test."""
    if not ORIGINAL_DB.exists():
        pytest.skip("data/soretrak_gtfs.db absente — lancer scripts/build_db.py d'abord")
    dest = tmp_path / "copy.db"
    shutil.copy(ORIGINAL_DB, dest)
    return dest


def test_simple_select_works(db_copy):
    result = run_query("SELECT COUNT(*) AS n FROM stops", db_copy)
    assert result.ok
    assert result.rows[0]["n"] == 192


def test_nonexistent_db_is_an_error_not_a_crash(tmp_path):
    result = run_query("SELECT 1", tmp_path / "absente.db")
    assert not result.ok
    assert "introuvable" in result.error.lower()


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO stops(stop_id) VALUES ('x')",
        "UPDATE stops SET stop_name='x'",
        "DELETE FROM stops",
        "DROP TABLE stops",
        "ALTER TABLE stops ADD COLUMN x TEXT",
        "PRAGMA table_info(stops)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "SELECT 1; DROP TABLE stops",
        "SELECT 1; SELECT 2",
        "WITH x AS (SELECT 1) DELETE FROM stops",
        "WITH x AS (SELECT 1) INSERT INTO stops(stop_id) SELECT 'y'",
        "REPLACE INTO stops(stop_id) VALUES ('z')",
        "VACUUM INTO '/tmp/leak.db'",
    ],
)
def test_dangerous_sql_is_blocked(db_copy, sql):
    result = run_query(sql, db_copy)
    assert not result.ok, f"aurait dû être bloqué : {sql!r}"


def test_integrity_unchanged_after_dangerous_attempts(db_copy):
    for sql in [
        "INSERT INTO stops(stop_id) VALUES ('x')",
        "DELETE FROM stops",
        "DROP TABLE stops",
    ]:
        run_query(sql, db_copy)
    conn = sqlite3.connect(db_copy)
    count = conn.execute("SELECT COUNT(*) FROM stops").fetchone()[0]
    conn.close()
    assert count == 192


@pytest.mark.parametrize(
    "sql",
    [
        "select count(*) as n from stops",
        "-- commentaire\nSELECT COUNT(*) AS n FROM stops",
        "SELECT COUNT(*) AS n FROM stops;",
        "SELECT 'DELETE' AS n",
    ],
)
def test_benign_variants_allowed(db_copy, sql):
    result = run_query(sql, db_copy)
    assert result.ok, f"aurait dû passer : {sql!r} ({result.error})"


def test_row_cap_and_truncated_flag(db_copy):
    result = run_query("SELECT * FROM shapes", db_copy, max_rows=10)
    assert result.ok
    assert result.row_count == 10
    assert result.truncated is True


def test_no_truncation_when_under_cap(db_copy):
    result = run_query("SELECT * FROM agency", db_copy, max_rows=50)
    assert result.ok
    assert result.truncated is False


def test_long_value_is_cut(db_copy):
    conn = sqlite3.connect(db_copy)
    conn.execute("PRAGMA query_only = OFF")
    conn.execute(
        "UPDATE stops SET stop_name = ? WHERE stop_id = (SELECT stop_id FROM stops LIMIT 1)",
        ("x" * 500,),
    )
    conn.commit()
    conn.close()
    result = run_query("SELECT stop_name FROM stops LIMIT 1", db_copy)
    assert result.ok
    assert len(result.rows[0]["stop_name"]) <= 201  # 200 + "…"


def test_query_timeout(db_copy):
    sql = """
        WITH RECURSIVE c(n) AS (
            SELECT 1 UNION ALL SELECT n + 1 FROM c LIMIT 100000000
        )
        SELECT COUNT(*) FROM c, stops, stops AS s2
    """
    start = time.monotonic()
    result = run_query(sql, db_copy, timeout_seconds=0.2)
    elapsed = time.monotonic() - start
    assert not result.ok
    assert elapsed < 3



def test_params_are_bound_not_pasted(db_copy):
    result = run_query("SELECT stop_name FROM stops WHERE stop_id = ?", db_copy, params=("432",))
    assert result.rows == [{"stop_name": "BAB JEDID"}]
    # an apostrophe in the value breaks nothing and injects nothing
    assert run_query("SELECT COUNT(*) AS n FROM stops WHERE stop_name = ?", db_copy,
                     params=("x' OR '1'='1",)).rows == [{"n": 0}]
