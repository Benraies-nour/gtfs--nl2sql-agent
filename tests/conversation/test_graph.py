"""Main graph and memory, with fake components (no network call)."""
import pytest

from soretrak.conversation import graph as g
from soretrak import llm
from soretrak.components import respond
from soretrak.contracts import AgentResult, Candidate, RouterDecision

BAB = Candidate(type="arret", id="432", nom="BAB JEDID", score=94, raison="")
BIR = Candidate(type="arret", id="579", nom="Bir Jdid", score=75, raison="")
REQ = {"q1": {"sql": "…", "columns": ["route_id"], "rows": [{"route_id": "21"}],
              "row_count": 1, "truncated": False, "error": None}}


class Fakes:
    """Fake Router / agent / Synthesizer driven by output queues."""

    def __init__(self, monkeypatch):
        self.decisions, self.agent_results = [], []
        self.router_calls, self.agent_calls, self.synth_calls = [], [], []
        monkeypatch.setattr(g.router, "route", self._route)
        monkeypatch.setattr(g, "run_sql_agent", self._agent)
        monkeypatch.setattr(g.synthesizer, "synthesize", self._synth)

    def _route(self, message, history=None, contexte=None, en_attente=None):
        self.router_calls.append({"message": message, "history": list(history or []),
                                  "contexte": list(contexte or []), "en_attente": en_attente})
        out = self.decisions.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def _agent(self, question, entites=None, lieux_resolus=None):
        self.agent_calls.append({"question": question, "entites": list(entites or []),
                                 "lieux": list(lieux_resolus or [])})
        result = self.agent_results.pop(0)
        return {"resultat": result, "requetes": REQ, "messages": [{}, {}], "nb_appels": 2}

    def _synth(self, question, result, requetes, graphique=False):
        self.synth_calls.append(question)
        return f"Synthèse : {question}"


@pytest.fixture
def fakes(monkeypatch):
    return Fakes(monkeypatch)


@pytest.fixture
def app():
    return g.build_graph()


def transport(q):
    return RouterDecision(intent="transport", question_autonome=q)


# --- paths by intent ---------------------------------------------------------

@pytest.mark.parametrize("intent", ["conversation", "trop_vague", "hors_perimetre", "hors_sujet"])
def test_non_transport_intents_answer_directly(fakes, app, intent):
    fakes.decisions = [RouterDecision(intent=intent, reponse=f"texte {intent}")]
    response = g.answer("message", "t1", graph=app)
    assert response.texte == f"texte {intent}"
    assert response.chemin == intent
    assert fakes.agent_calls == []


@pytest.mark.parametrize("statut", ["repondu", "aucune_donnee"])
def test_data_statuses_go_to_synthesizer(fakes, app, statut):
    fakes.decisions = [transport("Lignes à bab jdid")]
    fakes.agent_results = [AgentResult(statut=statut, preuves=["q1"])]
    response = g.answer("fema bus l bab jdid", "t1", graph=app)
    assert response.texte == "Synthèse : Lignes à bab jdid"
    assert response.chemin == statut
    assert response.details["agent"]["requetes"]["q1"]["rows"] == [{"route_id": "21"}]


def test_impossible_goes_to_respond_without_agent_note(fakes, app):
    fakes.decisions = [RouterDecision(intent="transport", question_autonome="Arrêts à Paris",
                                      lieux=["paris"])]
    fakes.agent_results = [AgentResult(statut="impossible",
                                       note="colonne trip_id absente de v_route_stops ; budget épuisé")]
    response = g.answer("bus à paris", "t1", graph=app)
    assert "« paris »" in response.texte            # place not found, given by the code
    for technical in ("trip_id", "v_route_stops", "budget", "colonne", "SQL"):
        assert technical not in response.texte
    assert response.details["agent"]["resultat"]["note"].startswith("colonne")   # kept for the trace
    assert fakes.synth_calls == []


def test_agent_echec_shows_fixed_message(fakes, app):
    fakes.decisions = [transport("x")]
    fakes.agent_results = [AgentResult(statut="echec", note="budget d'appels d'outils épuisé")]
    response = g.answer("x", "t1", graph=app)
    assert response.texte == respond.MESSAGE_ECHEC


def test_agent_rate_limit_shows_quota_message(fakes, app):
    fakes.decisions = [transport("x")]
    fakes.agent_results = [AgentResult(statut="echec", note="limite de débit", quota=True)]
    assert g.answer("x", "t1", graph=app).texte == respond.MESSAGE_QUOTA


def test_router_errors(fakes, app):
    fakes.decisions = [llm.LLMError("invalide"), llm.LLMRateLimitError("429")]
    assert g.answer("x", "t1", graph=app).texte == respond.MESSAGE_ECHEC
    assert g.answer("x", "t1", graph=app).texte == respond.MESSAGE_QUOTA


# --- memory -----------------------------------------------------------------

def test_history_is_passed_to_router(fakes, app):
    fakes.decisions = [transport("Arrêts de la ligne 21"), transport("Arrêts de la ligne 22")]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])] * 2
    g.answer("la ligne 21 passe par quels arrêts ?", "t1", graph=app)
    g.answer("et la 22 ?", "t1", graph=app)
    history = fakes.router_calls[1]["history"]
    assert history == [
        {"role": "user", "content": "la ligne 21 passe par quels arrêts ?"},
        {"role": "assistant", "content": "Synthèse : Arrêts de la ligne 21"},
    ]


def test_threads_are_isolated(fakes, app):
    fakes.decisions = [RouterDecision(intent="conversation", reponse="ok")] * 2
    g.answer("bonjour", "A", graph=app)
    g.answer("salut", "B", graph=app)
    assert fakes.router_calls[1]["history"] == []


def test_ambiguous_then_choice_confirms_entity_without_second_question(fakes, app):
    fakes.decisions = [transport("Prochain bus à bab jdid"), transport("Prochain bus à BAB JEDID")]
    fakes.agent_results = [
        AgentResult(statut="ambigu", clarification=[BAB, BIR]),
        AgentResult(statut="repondu", preuves=["q1"]),
    ]
    first = g.answer("prochain bus à bab jdid", "t1", graph=app)
    assert first.chemin == "ambigu"
    assert "BAB JEDID" in first.texte and "Bir Jdid" in first.texte

    second = g.answer("Bab Jedid", "t1", graph=app)
    assert second.chemin == "repondu"
    assert [c.id for c in fakes.agent_calls[1]["entites"]] == ["432"]
    assert second.details["entites_confirmees"][0]["nom"] == "BAB JEDID"


def test_pending_clarification_is_cleared_after_one_message(fakes, app):
    fakes.decisions = [transport("q"), RouterDecision(intent="conversation", reponse="ok"),
                       RouterDecision(intent="conversation", reponse="ok")]
    fakes.agent_results = [AgentResult(statut="ambigu", clarification=[BAB, BIR])]
    g.answer("prochain bus à bab jdid", "t1", graph=app)
    g.answer("merci", "t1", graph=app)           # not a choice
    response = g.answer("Bab Jedid", "t1", graph=app)  # too late
    assert response.details["entites_confirmees"] == []


def test_per_turn_fields_are_reset(fakes, app):
    fakes.decisions = [transport("q"), RouterDecision(intent="conversation", reponse="ok")]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])]
    g.answer("q", "t1", graph=app)
    response = g.answer("merci", "t1", graph=app)
    assert response.details["agent"] is None


# --- pre-retrieval: places resolved by the code before the agent ---------------

def test_router_places_are_resolved_before_agent(fakes, app):
    fakes.decisions = [RouterDecision(intent="transport", question_autonome="Lignes à bab jdid",
                                      lieux=["bab jdid"])]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])]
    g.answer("fema bus l bab jdid ?", "t1", graph=app)
    lieux = fakes.agent_calls[0]["lieux"]
    assert lieux[0]["texte"] == "bab jdid" and lieux[0]["statut"] == "trouve"
    assert lieux[0]["candidats"][0]["id"] == "432"
    assert "21" in lieux[0]["candidats"][0]["lignes"]


def test_no_places_means_empty_resolution(fakes, app):
    fakes.decisions = [transport("Combien de lignes ont des horaires ?")]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])]
    g.answer("combien de lignes ont des horaires ?", "t1", graph=app)
    assert fakes.agent_calls[0]["lieux"] == []


# --- memory layer: structured conversation context ----------------------------

SOUSSE_REQ = {"q1": {"sql": "…", "columns": ["route_id"],
                     "rows": [{"route_id": "311"}, {"route_id": "768"}, {"route_id": "790"}],
                     "row_count": 3, "truncated": False, "error": None}}


def test_context_of_previous_turn_reaches_router(fakes, app, monkeypatch):
    def agent(question, entites=None, lieux_resolus=None):
        fakes.agent_calls.append({"question": question, "lieux": list(lieux_resolus or [])})
        return {"resultat": fakes.agent_results.pop(0), "requetes": SOUSSE_REQ,
                "messages": [{}, {}], "nb_appels": 1}
    monkeypatch.setattr(g, "run_sql_agent", agent)
    fakes.decisions = [
        RouterDecision(intent="transport", question_autonome="Un bus pour Sousse ?", lieux=["sousse"]),
        RouterDecision(intent="transport", question_autonome="Horaires des lignes 311, 768, 790"),
    ]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"]),
                           AgentResult(statut="repondu", preuves=["q1"])]
    g.answer("un bus pour sousse ?", "t1", graph=app)
    g.answer("à quelle heure ?", "t1", graph=app)
    contexte = fakes.router_calls[1]["contexte"]
    assert contexte[-1]["question"] == "Un bus pour Sousse ?"
    assert contexte[-1]["lignes"] == ["311", "768", "790"]
    assert contexte[-1]["lieux"][0]["nom"] == "SOUSSE"


def test_context_keeps_last_three_transport_turns(fakes, app):
    fakes.decisions = [transport(f"question {i}") for i in range(5)] +         [RouterDecision(intent="conversation", reponse="ok")]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])] * 5
    for i in range(6):
        g.answer(f"message {i}", "t1", graph=app)
    contexte = fakes.router_calls[-1]["contexte"]
    assert [c["question"] for c in contexte] == ["question 2", "question 3", "question 4"]


# --- on-demand map --------------------------------------------------------------

def test_map_only_when_router_asks(fakes, app):
    fakes.decisions = [
        RouterDecision(intent="transport", question_autonome="Lignes à bab jdid", lieux=["bab jdid"]),
        RouterDecision(intent="transport", question_autonome="Trajet de bab jdid", lieux=["bab jdid"],
                       carte=True),
    ]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])] * 2
    assert g.answer("lignes à bab jdid", "t1", graph=app).details["carte"] is None
    carte = g.answer("montre-moi bab jdid sur la carte", "t1", graph=app).details["carte"]
    assert carte["points"][0]["nom"] == "BAB JEDID"



def test_chart_only_when_router_asks(fakes, app):
    fakes.decisions = [
        RouterDecision(intent="transport", question_autonome="Nombre de départs par ligne"),
        RouterDecision(intent="transport", question_autonome="Nombre de départs par ligne", graphique=True),
    ]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])] * 2
    assert g.answer("départs par ligne", "t1", graph=app).details["graphique"] is None
    # REQ has a single row: no chart possible, but no error
    assert g.answer("départs par ligne en barres", "t1", graph=app).details["graphique"] is None



# --- pending question after a Router clarification request ----------------------

def test_router_clarification_is_remembered_for_next_message(fakes, app):
    fakes.decisions = [
        RouterDecision(intent="trop_vague", reponse="Pour quel arrêt ou quelle ligne ?"),
        RouterDecision(intent="transport", question_autonome="Nombre de départs par heure de la ligne 3",
                       graphique=True),
        RouterDecision(intent="conversation", reponse="ok"),
    ]
    fakes.agent_results = [AgentResult(statut="repondu", preuves=["q1"])]
    g.answer("Visualise-moi le nombre de départs par heure.", "t1", graph=app)
    g.answer("Pour la ligne 3", "t1", graph=app)
    g.answer("merci", "t1", graph=app)
    assert fakes.router_calls[0]["en_attente"] is None
    assert fakes.router_calls[1]["en_attente"] == {
        "question": "Visualise-moi le nombre de départs par heure.",
        "demande": "Pour quel arrêt ou quelle ligne ?"}
    assert fakes.router_calls[2]["en_attente"] is None          # cleared once used


def test_successive_clarifications_accumulate(fakes, app):
    fakes.decisions = [RouterDecision(intent="trop_vague", reponse="Quelle ligne ?"),
                       RouterDecision(intent="trop_vague", reponse="Quel sens ?"),
                       RouterDecision(intent="conversation", reponse="ok")]
    for m in ("les horaires", "la 21", "aller"):
        g.answer(m, "t1", graph=app)
    assert fakes.router_calls[2]["en_attente"]["question"] == "les horaires / la 21"
