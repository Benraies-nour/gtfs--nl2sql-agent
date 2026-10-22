"""SQL Agent with a scripted fake LLM (no network call).

The tools, however, really run against the database.
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from soretrak.config import ROOT

from soretrak import llm

from soretrak.components import sql_agent

DB = ROOT / "data" / "soretrak_gtfs.db"


def call(name, **args):
    return (name, args)


def turn(*calls):
    """One agent turn: a message with one or more tool calls."""
    tool_calls = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    return SimpleNamespace(content="", tool_calls=tool_calls)


@pytest.fixture
def script(monkeypatch):
    if not DB.exists():
        pytest.skip("data/soretrak_gtfs.db absente")

    def install(*turns):
        remaining = list(turns)
        seen = []

        def fake_with_tools(messages, tools, model=None):
            seen.append(messages)
            outcome = remaining.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(llm, "with_tools", fake_with_tools)
        return seen

    return install


def run(question="question"):
    return sql_agent.run_sql_agent(question, now=datetime(2026, 10, 3, 8, 0), db_path=DB)


def last_tool_output(seen):
    return seen[-1][-1]["content"]




def test_normal_path_resolve_sql_finish(script):
    script(
        turn(call("resolve_place", texte="bab jdid")),
        turn(call("run_sql", sql="SELECT DISTINCT route_id FROM v_route_stops WHERE stop_id = '432'")),
        turn(call("finish", statut="repondu", preuves=["q1"])),
    )
    final = run("Quelles lignes passent par Bab Jedid ?")
    assert final["resultat"].statut == "repondu"
    assert final["resultat"].preuves == ["q1"]
    assert final["requetes"]["q1"]["row_count"] > 0
    assert final["nb_appels"] == 2


def test_resolve_place_output_reaches_agent(script):
    seen = script(
        turn(call("resolve_place", texte="bab jdid")),
        turn(call("finish", statut="impossible", note="test")),
    )
    run()
    output = json.loads(last_tool_output(seen))
    assert output["statut"] == "trouve"
    assert output["candidats"][0]["id"] == "432"


def test_sql_error_then_correction(script):
    seen = script(
        turn(call("run_sql", sql="SELECT nom FROM stops")),
        turn(call("run_sql", sql="SELECT COUNT(*) AS n FROM stops")),
        turn(call("finish", statut="repondu", preuves=["q2"])),
    )
    final = run()
    assert "no such column" in seen[1][-1]["content"]
    assert final["resultat"].statut == "repondu"


def test_failed_query_cannot_be_evidence(script):
    seen = script(
        turn(call("run_sql", sql="SELECT nom FROM stops")),
        turn(call("finish", statut="repondu", preuves=["q1"])),
        turn(call("finish", statut="impossible", note="abandon")),
    )
    final = run()
    assert "ont échoué" in last_tool_output(seen)
    assert final["resultat"].statut == "impossible"


def test_zero_rows_gives_aucune_donnee(script):
    script(
        turn(call("run_sql", sql="SELECT departure_time FROM v_departures WHERE route_id = '5'")),
        turn(call("finish", statut="aucune_donnee", preuves=["q1"])),
    )
    final = run("Horaires de la ligne 5")
    assert final["requetes"]["q1"]["row_count"] == 0
    assert final["resultat"].statut == "aucune_donnee"


def test_ambiguous_place(script):
    candidats = [
        {"type": "arret", "id": "579", "nom": "Bir Jdid"},
        {"type": "arret", "id": "227", "nom": "BIR JEDID-Bouhajla"},
    ]
    script(
        turn(call("resolve_place", texte="bir jedid")),
        turn(call("finish", statut="ambigu", clarification=candidats)),
    )
    final = run()
    assert final["resultat"].statut == "ambigu"
    assert [c.id for c in final["resultat"].clarification] == ["579", "227"]


def test_ambigu_without_candidates_is_refused(script):
    seen = script(
        turn(call("finish", statut="ambigu")),
        turn(call("finish", statut="impossible")),
    )
    run()
    assert "candidats" in last_tool_output(seen)


def test_nonexistent_evidence_is_refused(script):
    seen = script(
        turn(call("finish", statut="repondu", preuves=["q7"])),
        turn(call("finish", statut="impossible")),
    )
    final = run()
    assert "inexistantes" in last_tool_output(seen)
    assert final["resultat"].statut == "impossible"


def test_agent_cannot_choose_echec(script):
    seen = script(
        turn(call("finish", statut="echec")),
        turn(call("finish", statut="impossible")),
    )
    run()
    assert "Statut invalide" in last_tool_output(seen)


def test_budget_exhausted(script):
    sql = call("run_sql", sql="SELECT 1")
    seen = script(*[turn(sql)] * (sql_agent.TOOL_BUDGET + 1))
    final = run()
    assert final["resultat"].statut == "echec"
    assert final["nb_appels"] == sql_agent.TOOL_BUDGET
    assert "dernier appel" in seen[-1][-1]["content"]


def test_budget_counts_parallel_calls(script):
    script(turn(*[call("run_sql", sql="SELECT 1")] * (sql_agent.TOOL_BUDGET + 1)))
    assert run()["resultat"].statut == "echec"


def test_finish_after_last_budgeted_call_is_allowed(script):
    script(
        *[turn(call("run_sql", sql="SELECT 1"))] * sql_agent.TOOL_BUDGET,
        turn(call("finish", statut="repondu", preuves=[f"q{sql_agent.TOOL_BUDGET}"])),
    )
    assert run()["resultat"].statut == "repondu"


def test_llm_error_gives_echec(script):
    script(llm.LLMError("panne"))
    final = run()
    assert final["resultat"].statut == "echec"


def test_invalid_json_arguments_are_reported(script):
    bad = SimpleNamespace(content="", tool_calls=[
        SimpleNamespace(id="c0", function=SimpleNamespace(name="run_sql", arguments="{pas du json"))
    ])
    seen = script(bad, turn(call("finish", statut="impossible")))
    run()
    assert "Arguments invalides" in last_tool_output(seen)


def test_max_turns(script):
    script(*[turn(call("finish", statut="ambigu"))] * (sql_agent.MAX_TURNS + 1))
    assert run()["resultat"].statut == "echec"


def test_system_prompt_is_filled():
    prompt = sql_agent.build_system_prompt([], now=datetime(2026, 10, 3, 8, 15))
    assert "08:15:00" in prompt
    assert "v_departures" in prompt          # schema
    assert "Manufacture des tabacs" in prompt  # examples
    assert "Prix des tickets" in prompt      # scope sheet
    assert "{{" not in prompt


# --- pre-retrieval ----------------------------------------------------------

def _lieux():
    from soretrak.places.facts import resolve_mentions
    return resolve_mentions(["laminia mansoura", "bab jdid"], db_path=DB)


def test_facts_are_in_prompt_and_registered_as_q0(script):
    seen = script(turn(call("finish", statut="aucune_donnee", preuves=["q0"])))
    final = sql_agent.run_sql_agent("Route LAMINIA MANSOURA", now=datetime(2026, 10, 3, 8, 0),
                                    db_path=DB, lieux_resolus=_lieux())
    prompt = seen[0][1]["content"]           # agent message: the question and its facts
    assert prompt.startswith("Route LAMINIA MANSOURA")
    assert "ligne 1 « LAMINIA ESSAIED MANSOURA »" in prompt and "horaires : non" in prompt
    assert "stop_id '432'" in prompt
    assert final["resultat"].statut == "aucune_donnee"
    assert final["requetes"]["q0"]["rows"][0]["nom"] == "LAMINIA ESSAIED MANSOURA"
    assert final["nb_appels"] == 0


def test_first_query_is_q1_even_with_q0(script):
    script(
        turn(call("run_sql", sql="SELECT 1")),
        turn(call("finish", statut="repondu", preuves=["q1"])),
    )
    final = sql_agent.run_sql_agent("x", now=datetime(2026, 10, 3, 8, 0), db_path=DB,
                                    lieux_resolus=_lieux())
    assert set(final["requetes"]) == {"q0", "q1"}


def test_without_places_the_message_is_the_question_alone():
    assert sql_agent.build_user_message("Combien de lignes ?", []) == "Combien de lignes ?"



def test_impossible_after_budget_exhausted_becomes_echec(script):
    script(
        *[turn(call("run_sql", sql="SELECT nom FROM stops"))] * sql_agent.TOOL_BUDGET,
        turn(call("finish", statut="impossible", note="colonne absente ; budget épuisé")),
    )
    result = run()["resultat"]
    assert result.statut == "echec"
    assert "budget" in result.note


def test_impossible_within_budget_stays_impossible(script):
    script(turn(call("finish", statut="impossible", note="lieu introuvable : paris")))
    assert run()["resultat"].statut == "impossible"



def test_rate_limit_sets_quota_flag(script):
    script(llm.LLMRateLimitError("429"))
    result = run()["resultat"]
    assert result.statut == "echec" and result.quota is True


def test_other_llm_error_is_not_quota(script):
    script(llm.LLMError("panne"))
    assert run()["resultat"].quota is False


# --- signals on suspicious results --------------------------------------------

def _observe(sql):
    from soretrak.components.tools import run_run_sql
    return json.loads(run_run_sql({"sql": sql}, {}, DB))


TRAJET = ("FROM stop_times a JOIN stop_times b ON a.trip_id = b.trip_id "
          "WHERE a.stop_id = '429' AND b.stop_id = '432' AND a.stop_sequence < b.stop_sequence")


def test_text_time_subtraction_is_flagged():
    obs = _observe(f"SELECT b.departure_time - a.departure_time AS duree_min {TRAJET}")
    assert any("Soustraction d'heures en texte" in w for w in obs["attention"])
    assert any("Durée nulle ou négative" in w for w in obs["attention"])


def test_correct_duration_has_no_warning():
    obs = _observe("SELECT (strftime('%s', b.departure_time) - strftime('%s', a.departure_time)) / 60 "
                   f"AS duree_min {TRAJET}")
    assert obs["lignes"][0]["duree_min"] == 16
    assert "attention" not in obs


def test_empty_result_is_not_flagged():
    # An empty result is often the right answer ("no direct line"): no signal.
    obs = _observe("SELECT departure_time FROM v_departures WHERE route_id = '21' AND direction_id = 1")
    assert obs["nb_lignes"] == 0 and "attention" not in obs
