"""Streamlit interface, with a fake answer function (no network call)."""
import pytest
from streamlit.testing.v1 import AppTest

from soretrak.conversation import graph
from soretrak.contracts import Response

AGENT = {
    "resultat": {"statut": "repondu", "preuves": ["q1"], "note": None, "clarification": []},
    "requetes": {"q1": {"sql": "SELECT 1", "columns": ["route_id"], "rows": [{"route_id": "21"}],
                        "row_count": 1, "truncated": False, "error": None}},
    "trace": [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function",
             "function": {"name": "run_sql", "arguments": '{"sql": "SELECT route_id FROM routes"}'}}]},
        {"role": "tool", "tool_call_id": "c0",
         "content": '{"id": "q1", "colonnes": ["route_id"], "lignes": [], "nb_lignes": 1, "tronque": false}'},
    ],
    "nb_appels": 1,
}
DECISION = {"intent": "transport", "question_autonome": "Lignes à bab jdid", "reponse": None}
CANDIDATS = [
    {"type": "arret", "id": "432", "nom": "BAB JEDID", "score": 0, "raison": ""},
    {"type": "arret", "id": "579", "nom": "Bir Jdid", "score": 0, "raison": ""},
]


@pytest.fixture
def calls(monkeypatch):
    received = []

    def fake_answer(message, thread_id):
        received.append((message, thread_id))
        if message == "ambigu":
            agent = {**AGENT, "resultat": {"statut": "ambigu", "preuves": [], "note": None,
                                           "clarification": CANDIDATS}}
            return Response(texte="Parlez-vous de BAB JEDID ou Bir Jdid ?", chemin="ambigu",
                            details={"decision": DECISION, "agent": agent})
        if message == "carte":
            carte = {"titre": "Ligne 22 : A → B", "trace": "gps", "note": None,
                     "chemin": [[10.08, 35.65], [10.10, 35.67]],
                     "points": [{"nom": "A", "lat": 35.65, "lon": 10.08, "role": "depart"},
                                {"nom": "B", "lat": 35.67, "lon": 10.10, "role": "arrivee"}]}
            return Response(texte="Voici le trajet.", chemin="repondu",
                            details={"decision": DECISION, "agent": AGENT, "carte": carte})
        if message == "graphique":
            graphique = {"titre": "Nombre d'arrêts par ligne", "axe_libelle": "Ligne",
                         "axe_valeur": "Nombre d'arrêts", "orientation": "horizontale",
                         "donnees": [{"libelle": "Ligne 21", "valeur": 19},
                                     {"libelle": "Ligne 22", "valeur": 17}],
                         "tronque": False, "note": None}
            return Response(texte="Voici le graphique.", chemin="repondu",
                            details={"decision": DECISION, "agent": AGENT, "graphique": graphique})
        if message == "panne":
            raise RuntimeError("Traceback technique")
        return Response(texte=f"Réponse à : {message}", chemin="repondu",
                        details={"decision": DECISION, "agent": AGENT})

    monkeypatch.setattr(graph, "answer", fake_answer)
    return received


def run_app():
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    return at




def test_question_answer_and_how_panel(calls):
    at = run_app()
    at.chat_input[0].set_value("lignes à bab jdid").run()
    assert not at.exception
    texts = [m.value for m in at.markdown]
    assert "Réponse à : lignes à bab jdid" in texts
    assert at.expander[0].label == "Comment j'ai trouvé"
    panel = " ".join(m.value for m in at.expander[0].markdown)
    assert "Question comprise :" in panel and "Lignes à bab jdid" in panel
    assert "horaires théoriques SORETRAK" in panel
    for technical in ("SQL", "q1", "Preuve", "repondu", "transport", "route_id"):
        assert technical not in panel, technical
    assert len(at.code) == 0                       # no SQL outside debug mode
    assert len(at.dataframe) == 1
    assert list(at.dataframe[0].value.columns) == ["Ligne"]   # route_id → Ligne


def test_debug_mode_shows_sql(calls):
    at = AppTest.from_file("app.py", default_timeout=10)
    at.query_params["debug"] = "1"
    at.run()
    at.chat_input[0].set_value("lignes à bab jdid").run()
    assert at.code[0].value == "SELECT route_id FROM routes"
    assert any("intent : transport" in c.value for c in at.caption)


def test_same_thread_for_the_whole_session(calls):
    at = run_app()
    at.chat_input[0].set_value("a").run()
    at.chat_input[0].set_value("b").run()
    assert calls[0][1] == calls[1][1]


def test_ambiguous_candidates_are_clickable(calls):
    at = run_app()
    at.chat_input[0].set_value("ambigu").run()
    next(b for b in at.button if b.label == "Bir Jdid").click().run()
    assert calls[-1][0] == "Bir Jdid"


def test_unexpected_error_shows_fixed_message(calls):
    at = run_app()
    at.chat_input[0].set_value("panne").run()
    assert not at.exception
    assert any("reformuler" in w.value for w in at.warning)
    assert not any("Traceback" in m.value for m in at.markdown)


def test_new_conversation_changes_thread(calls):
    at = run_app()
    at.chat_input[0].set_value("a").run()
    next(b for b in at.button if b.label == "Nouvelle conversation").click().run()
    at.chat_input[0].set_value("b").run()
    assert calls[0][1] != calls[1][1]



def test_map_is_shown_only_when_present(calls):
    at = run_app()
    at.chat_input[0].set_value("lignes").run()
    assert len(at.get("iframe")) == 0
    at.chat_input[0].set_value("carte").run()
    assert not at.exception
    assert len(at.get("iframe")) == 1
    assert any("Ligne 22 : A → B" in m.value for m in at.markdown)
    assert any("Tracé GPS : 2 points" in c.value for c in at.caption)



def test_chart_is_shown_only_when_present(calls):
    at = run_app()
    at.chat_input[0].set_value("lignes").run()
    assert len(at.get("arrow_vega_lite_chart")) == 0
    at.chat_input[0].set_value("graphique").run()
    assert not at.exception
    assert len(at.get("arrow_vega_lite_chart")) == 1
    assert any("Nombre d'arrêts par ligne" in m.value for m in at.markdown)


def test_welcome_screen_shows_the_title_and_no_button(calls):
    at = run_app()
    assert not at.exception
    assert len(at.sidebar) == 0
    assert any("Asslema, je suis ton assistant" in m.value for m in at.markdown)
    assert len(at.button) == 0          # no suggestions, no "Nouvelle conversation" yet


def test_logo_header_always_visible(calls):
    at = run_app()
    header = next(m.value for m in at.markdown if 'class="sa-header"' in m.value)
    assert header.count("data:image/png;base64,") == 2
    assert header.index('alt="République tunisienne') < header.index('alt="SORETRAK"')   # Tunisian Republic logo on the left
    at.chat_input[0].set_value("lignes").run()
    assert any('class="sa-header"' in m.value for m in at.markdown)
    assert not any("Asslema" in m.value for m in at.markdown)        # welcome screen replaced by the conversation



def test_background_image_on_welcome_only(calls):
    at = run_app()
    assert any("data:image/jpeg;base64," in m.value for m in at.markdown)
    at.chat_input[0].set_value("lignes").run()
    assert not any("data:image/jpeg;base64," in m.value for m in at.markdown)
