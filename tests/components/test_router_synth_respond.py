"""Router, Synthesizer and Respond with a fake LLM (no network call)."""
import pytest

from soretrak import llm

from soretrak.components import respond, router, synthesizer
from soretrak.contracts import AgentResult, Candidate, RouterDecision


@pytest.fixture
def fake_structured(monkeypatch):
    """Replace llm.structured: keep the prompts received, return the planned output."""
    calls = []

    def install(output):
        def fake(system, user, schema, model=None):
            calls.append({"system": system, "user": user, "schema": schema, "model": model})
            if isinstance(output, Exception):
                raise output
            return schema.model_validate(output)
        monkeypatch.setattr(llm, "structured", fake)
        return calls

    return install


# --- Router -----------------------------------------------------------------

def test_router_prompt_contains_rules_and_perimeter():
    prompt = router.build_system_prompt()
    for expected in ["transport", "trop_vague", "hors_perimetre", "hors_sujet", "conversation",
                     "ne les traduis jamais", "Prix des tickets", "une seule question courte"]:
        assert expected in prompt
    assert "{{" not in prompt


def test_router_returns_decision(fake_structured):
    calls = fake_structured({"intent": "transport", "question_autonome": "Lignes à bab jdid"})
    decision = router.route("fema bus l bab jdid ?")
    assert decision == RouterDecision(intent="transport", question_autonome="Lignes à bab jdid")
    assert calls[0]["schema"] is RouterDecision


def test_router_keeps_last_three_exchanges(fake_structured):
    calls = fake_structured({"intent": "transport", "question_autonome": "Horaires de la ligne 22"})
    history = []
    for i in range(5):
        history += [{"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"réponse {i}"}]
    router.route("et pour la 22 ?", history)
    user = calls[0]["user"]
    assert "question 1" not in user
    assert "question 2" in user and "réponse 4" in user
    assert user.rstrip().endswith("et pour la 22 ?")


def test_router_first_message_has_no_history(fake_structured):
    calls = fake_structured({"intent": "conversation", "reponse": "Bonjour !"})
    router.route("asslema")
    assert "début de la conversation" in calls[0]["user"]


def test_router_error_propagates(fake_structured):
    fake_structured(llm.LLMError("sortie invalide"))
    with pytest.raises(llm.LLMError):
        router.route("x")


# --- Synthesizer ------------------------------------------------------------

REQUETES = {
    "q1": {"sql": "…", "columns": ["route_id"], "rows": [{"route_id": "21"}, {"route_id": "22"}],
           "row_count": 2, "truncated": False, "error": None},
    "q2": {"sql": "…", "columns": ["n"], "rows": [{"n": 0}],
           "row_count": 1, "truncated": True, "error": None},
    "q3": {"sql": "…", "columns": ["x"], "rows": [{"x": "NE DOIT PAS APPARAÎTRE"}],
           "row_count": 1, "truncated": False, "error": None},
}


def test_synthesizer_receives_only_evidence_rows(fake_structured):
    calls = fake_structured({"reponse": "  Les lignes 21 et 22 passent par Bab Jedid.  "})
    result = AgentResult(statut="repondu", preuves=["q1", "q2"], note="prix non disponible")
    text = synthesizer.synthesize("Lignes à bab jdid ?", result, REQUETES)
    assert text == "Les lignes 21 et 22 passent par Bab Jedid."
    user = calls[0]["user"]
    assert '"route_id": "21"' in user
    assert "TRONQUÉ" in user
    assert "prix non disponible" in user
    assert "NE DOIT PAS APPARAÎTRE" not in user


def test_synthesizer_prompt_rules():
    from soretrak.config import PROMPTS_DIR
    prompt = (PROMPTS_DIR / "synthesizer.md").read_text(encoding="utf-8")
    assert "N'affirme **que** ce qui est dans les résultats" in prompt
    assert "aucune_donnee" in prompt


# --- Respond ----------------------------------------------------------------

def _cand(id_, nom, type_="arret"):
    return Candidate(type=type_, id=id_, nom=nom, score=0, raison="")


def test_respond_ambigu_lists_candidates():
    result = AgentResult(statut="ambigu",
                         clarification=[_cand("432", "BAB JEDID"), _cand("579", "Bir Jdid")])
    assert respond.for_agent_result(result) == \
        "Plusieurs arrêts correspondent. Parlez-vous de BAB JEDID ou Bir Jdid ?"


def test_respond_ambigu_single_and_lines():
    assert respond.ambigu(AgentResult(statut="ambigu", clarification=[_cand("1", "SOUSSE")])) == \
        "Parlez-vous de SOUSSE ?"
    lines = AgentResult(statut="ambigu", clarification=[_cand("311", "A", "ligne"),
                                                       _cand("312", "B", "ligne")])
    assert respond.ambigu(lines).startswith("Plusieurs lignes")


def test_respond_impossible_never_shows_agent_note():
    result = AgentResult(statut="impossible", note="Requête SQL impossible (colonne trip_id absente)")
    text = respond.for_agent_result(result)
    assert "SQL" not in text and "trip_id" not in text and "colonne" not in text
    assert "+216 77 226 321" in text
    lieux = [{"texte": "paris", "statut": "introuvable", "candidats": []},
             {"texte": "bab jdid", "statut": "trouve", "candidats": [{}]}]
    assert "« paris »" in respond.for_agent_result(result, lieux)
    assert "bab jdid" not in respond.for_agent_result(result, lieux)


def test_respond_echec_shows_no_technical_detail():
    text = respond.for_agent_result(
        AgentResult(statut="echec", note="erreur LLM : Error code: 400 tool_use_failed"))
    assert text == respond.MESSAGE_ECHEC
    assert "400" not in text and "LLM" not in text


def test_respond_quota():
    assert "réessayer" in respond.quota()







def test_router_prompt_includes_conversation_context(fake_structured):
    calls = fake_structured({"intent": "transport", "question_autonome": "Horaires des lignes 311, 768, 790"})
    contexte = [{"question": "Un bus pour Sousse ?", "statut": "aucune_donnee",
                 "lieux": [{"type": "arret", "id": "257", "nom": "SOUSSE"}],
                 "lignes": ["311", "768", "790"], "arrets": []}]
    router.route("à quelle heure ?", [], contexte)
    user = calls[0]["user"]
    assert "Tour précédent : « Un bus pour Sousse ? »" in user
    assert "SOUSSE (arrêt 257)" in user and "lignes trouvées : 311, 768, 790" in user
    assert user.rstrip().endswith("à quelle heure ?")


def test_router_prompt_without_context():
    assert router.format_context([]) == "(aucun)"



def test_synthesizer_is_told_when_a_chart_is_shown(fake_structured):
    calls = fake_structured({"reponse": "La ligne 360 a le plus d'arrêts."})
    result = AgentResult(statut="repondu", preuves=["q1"])
    synthesizer.synthesize("Nombre d'arrêts par ligne", result, REQUETES, graphique=True)
    synthesizer.synthesize("Nombre d'arrêts par ligne", result, REQUETES)
    assert synthesizer.CHART_NOTE in calls[0]["user"]
    assert synthesizer.CHART_NOTE not in calls[1]["user"]








def test_router_prompt_includes_pending_question(fake_structured):
    calls = fake_structured({"intent": "transport", "question_autonome": "Départs par heure de la ligne 3"})
    router.route("Pour la ligne 3", [], [], {"question": "Visualise-moi le nombre de départs par heure.",
                                             "demande": "Pour quelle ligne ?"})
    user = calls[0]["user"]
    assert "Question en attente de précision : « Visualise-moi le nombre de départs par heure. »" in user
    assert user.index("Question en attente") < user.index("Dernier message")
    assert "Question en attente" not in router.format_user_prompt("x")
