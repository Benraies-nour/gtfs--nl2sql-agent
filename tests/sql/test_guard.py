"""Validation of the SQL shape, without touching a database."""
from soretrak.sql.guard import clean_sql, validate_sql


def check(sql: str) -> str | None:
    return validate_sql(clean_sql(sql))


def test_select_ok():
    assert check("SELECT 1") is None


def test_with_ok():
    assert check("WITH x AS (SELECT 1) SELECT * FROM x") is None


def test_lowercase_select_ok():
    assert check("select count(*) as n from stops") is None


def test_empty_rejected():
    assert check("") is not None
    assert check("   ") is not None


def test_insert_rejected():
    assert check("INSERT INTO stops(stop_id) VALUES ('x')") is not None


def test_update_rejected():
    assert check("UPDATE stops SET stop_name='x'") is not None


def test_delete_rejected():
    assert check("DELETE FROM stops") is not None


def test_drop_rejected():
    assert check("DROP TABLE stops") is not None


def test_alter_rejected():
    assert check("ALTER TABLE stops ADD COLUMN x TEXT") is not None


def test_pragma_rejected():
    assert check("PRAGMA table_info(stops)") is not None


def test_attach_rejected():
    assert check("ATTACH DATABASE '/tmp/x.db' AS x") is not None


def test_with_delete_rejected():
    # starts with WITH but the CTE feeds a DELETE: the shape alone cannot
    # see it, the authorizer (in sql/executor.py) is what actually blocks it.
    # Here we only check that the shape does not reject WITH.
    assert check("WITH x AS (SELECT 1) DELETE FROM stops") is None


def test_trailing_semicolon_stripped():
    assert check("SELECT 1;") is None


def test_leading_line_comment_stripped():
    assert check("-- compte les arrêts\nSELECT COUNT(*) FROM stops") is None


def test_leading_block_comment_stripped():
    assert check("/* compte les arrêts */\nSELECT COUNT(*) FROM stops") is None


def test_multiple_leading_comments_stripped():
    assert check("-- un\n-- deux\nSELECT 1") is None


def test_string_literal_with_forbidden_word_not_blocked_by_form():
    # "DELETE" appears in a string literal: the shape must not be
    # fooled by a keyword inside a text.
    assert check("SELECT 'DELETE' AS mot") is None
