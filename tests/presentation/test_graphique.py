"""On-demand chart: built by the code from the real rows of the evidence."""
from pathlib import Path

import pytest

from soretrak.config import DB_PATH
from soretrak.sql.executor import run_query
from soretrak.presentation.graphique import build_chart


def agent_for(*sqls):
    if not Path(DB_PATH).exists():
        pytest.skip("base absente")
    requetes = {f"q{i}": {"sql": s, **run_query(s, DB_PATH).model_dump()} for i, s in enumerate(sqls, 1)}
    return {"resultat": {"preuves": list(requetes), "statut": "repondu"}, "requetes": requetes}


def test_bars_per_line_sorted_by_value():
    c = build_chart("Nombre d'arrêts par ligne", agent_for(
        "SELECT route_id, COUNT(DISTINCT stop_id) AS nb_arrets FROM v_route_stops GROUP BY route_id"))
    assert c["orientation"] == "horizontale" and c["axe_valeur"] == "Nombre d'arrêts"
    assert len(c["donnees"]) == 34 and c["note"] is None
    values = [d["valeur"] for d in c["donnees"]]
    assert values == sorted(values, reverse=True)
    assert c["donnees"][0]["libelle"].startswith("Ligne ")


def test_columns_per_hour_in_chronological_order():
    c = build_chart("Départs par heure à Bab Jedid", agent_for(
        "SELECT substr(departure_time, 1, 2) || 'h' AS heure, COUNT(*) AS nb_departs "
        "FROM v_departures WHERE stop_id = '432' GROUP BY heure"))
    labels = [d["libelle"] for d in c["donnees"]]
    assert c["orientation"] == "verticale" and labels == sorted(labels)


def test_uses_last_suitable_evidence_and_skips_q0():
    agent = agent_for("SELECT COUNT(*) AS n FROM stops",
                      "SELECT route_id, COUNT(*) AS nb FROM v_departures GROUP BY route_id")
    agent["requetes"]["q0"] = {"rows": [{"a": "x", "n": 1}, {"a": "y", "n": 2}]}
    agent["resultat"]["preuves"].insert(0, "q0")
    assert build_chart("q", agent)["axe_valeur"] == "Nombre"


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) AS n FROM stops",                   # a single value
    "SELECT stop_name FROM stops LIMIT 5",               # no number
    "SELECT route_id, stop_sequence FROM v_route_stops LIMIT 5",   # numeric but not a measure
])
def test_no_chart_when_data_does_not_fit(sql):
    assert build_chart("q", agent_for(sql)) is None


def test_no_agent_no_chart():
    assert build_chart("q", None) is None





def test_bare_hour_numbers_get_h_suffix():
    c = build_chart("q", agent_for(
        "SELECT substr(departure_time, 1, 2) AS heure, COUNT(*) AS nb_departs "
        "FROM v_departures WHERE stop_id = '432' GROUP BY heure"))
    assert all(d["libelle"].endswith("h") for d in c["donnees"])
    assert c["orientation"] == "verticale"
