"""Conversation memory: context, clarification, confirmed entities."""
import pytest

from soretrak.contracts import Candidate
from soretrak.conversation import memory

BAB = Candidate(type="arret", id="432", nom="BAB JEDID", score=94, raison="")
BIR = Candidate(type="arret", id="579", nom="Bir Jdid", score=75, raison="")


# --- matching the reply to a clarification -----------------------------------

@pytest.mark.parametrize("message, expected", [
    ("Bab Jedid", "432"), ("bab jdid", "432"), ("BAB JEDID", "432"),
    ("Bir Jdid", "579"), ("1", "432"), ("le 2", "579"),
])
def test_match_clarification(message, expected):
    assert memory.match_clarification(message, [BAB, BIR]).id == expected


@pytest.mark.parametrize("message", ["merci", "3", "horaires de la ligne 21"])
def test_match_clarification_no_match(message):
    assert memory.match_clarification(message, [BAB, BIR]) is None



def test_turn_context_reads_q0_facts_and_stop_names():
    agent = {"resultat": {"preuves": ["q0", "q1"], "statut": "repondu"},
             "requetes": {"q0": {"rows": [{"lignes": "3, 21, 22"}]},
                          "q1": {"rows": [{"stop_name": "BAB JEDID"}, {"stop_name": "ERRAHBA"}]}}}
    ctx = memory.turn_context("q", [{"texte": "bab jdid", "statut": "trouve",
                               "candidats": [{"type": "arret", "id": "432", "nom": "BAB JEDID"}]}], agent)
    assert ctx["lignes"] == ["3", "21", "22"]
    assert ctx["arrets"] == ["BAB JEDID", "ERRAHBA"]
    assert ctx["lieux"] == [{"type": "arret", "id": "432", "nom": "BAB JEDID"}]


def test_push_context_keeps_last_turns():
    contexte = []
    for i in range(5):
        contexte = memory.push_context(contexte, {"question": f"q{i}"})
    assert [c["question"] for c in contexte] == ["q2", "q3", "q4"]


def test_confirm_adds_without_duplicate():
    entites = memory.confirm([], BAB)
    entites = memory.confirm(entites, BIR)
    entites = memory.confirm(entites, BAB)
    assert [e["id"] for e in entites] == ["579", "432"]
