"""SORETRAK Assist Streamlit interface.

    streamlit run app.py
"""
import base64
import json
import logging
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from soretrak.components import respond
from soretrak.presentation.carte import to_leaflet_html
from soretrak.presentation.formatting import keep_line_breaks, readable_rows
from soretrak.presentation.graphique import bar_chart
from soretrak.conversation.graph import answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="SORETRAK Assist", page_icon="🚌",
                   initial_sidebar_state="collapsed")


ASSETS = Path(__file__).parent / "assets"


@st.cache_resource
def logo_html(filename: str, alt: str) -> str:
    """A logo from assets/ embedded in the page (base64), empty if the file is missing."""
    path = ASSETS / filename
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="logo" src="data:image/png;base64,{data}" alt="{alt}">'


WELCOME_VEIL = 0 # white veil over the (already light) background image: keeps the text readable


@st.cache_resource
def welcome_background_css() -> str:
    """Welcome-screen background (assets/backgroundf.jpg): the whole background under the header,
    anchored at the bottom (bus and robot), with a white veil."""
    path = ASSETS / "backgroundf.jpg"
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    veil = f"rgba(255, 255, 255, {WELCOME_VEIL})"
    return (
        "<style>"
        # :not(:has(…)): the background disappears with the first bubble, without waiting for the answer
        '.stApp:not(:has([data-testid="stChatMessage"])) { background:'
        f" linear-gradient({veil}, {veil}),"
        f" url('data:image/jpeg;base64,{data}') no-repeat center 70% / cover fixed,"
        " #fff; }"
        '[data-testid="stBottom"] > div { background: transparent; }'
        "</style>"
    )


def new_conversation() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.history = []
    st.session_state.pending = None


if "thread_id" not in st.session_state:
    new_conversation()


# --- Display ------------------------------------------------------------------

STATUS_TEXT = {
    "repondu": "Réponse trouvée dans les horaires théoriques SORETRAK.",
    "aucune_donnee": "Les données disponibles ne contiennent pas cette information.",
    "ambigu": "Plusieurs lieux correspondent : une précision est nécessaire.",
    "impossible": "Cette information n'a pas été trouvée.",
    "echec": "La recherche n'a pas abouti.",
}


def debug_mode() -> bool:
    """Technical details (SQL, agent trace) only with ?debug=1 in the URL."""
    try:
        return st.query_params.get("debug") == "1"
    except Exception:
        return False


def show_how(details: dict) -> None:
    """The "Comment j'ai trouvé" panel, in plain language; technical details in debug mode."""
    decision = details.get("decision") or {}
    agent = details.get("agent")
    if not agent:      # conversation, refusal: nothing to show
        return
    result = agent["resultat"]
    with st.expander("Comment j'ai trouvé"):
        if decision.get("question_autonome"):
            st.markdown(f"**Question comprise :** {decision['question_autonome']}")
        lieux = [l for l in details.get("lieux") or [] if l["statut"] == "trouve" and l["candidats"]]
        if lieux:
            st.markdown("**Lieux reconnus :** " + " ; ".join(
                f"« {l['texte']} » → {'ligne' if l['candidats'][0]['type'] == 'ligne' else 'arrêt'} "
                f"{l['candidats'][0]['nom']}" for l in lieux))
        nb = sum(1 for q in result["preuves"] if q != "q0")
        st.markdown(f"**Données consultées :** horaires théoriques SORETRAK"
                    + (f", {nb} recherche{'s' if nb > 1 else ''}." if nb else "."))
        st.markdown(f"**Résultat :** {STATUS_TEXT.get(result['statut'], '')}")
        for qid in result["preuves"]:
            rows = agent["requetes"][qid]["rows"]
            if qid != "q0" and rows:
                st.dataframe(readable_rows(rows), hide_index=True)
        if debug_mode():
            show_trace(agent, decision)


def show_trace(agent: dict, decision: dict) -> None:
    """Developer mode: Router intent, tool calls, SQL and raw evidence."""
    st.divider()
    st.caption("Détails techniques (mode debug)")
    st.caption(f"intent : {decision.get('intent')} | lieux : {decision.get('lieux')} | "
               f"carte : {decision.get('carte')} | graphique : {decision.get('graphique')}")
    for message in agent["trace"]:
        if message.get("role") == "assistant":
            for call in message["tool_calls"]:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "run_sql":
                    st.code(args.get("sql", ""), language="sql")
                else:
                    st.caption(f"{name} {json.dumps(args, ensure_ascii=False)}")
        elif message.get("role") == "tool":
            st.caption(message["content"][:300])
    result = agent["resultat"]
    st.caption(f"statut : {result['statut']} | preuves : {result['preuves']} | "
               f"note : {result.get('note')} | appels : {agent['nb_appels']}")


def show_map(carte: dict | None) -> None:
    """On-demand map (Leaflet): GPS route points joined, departure, arrival, stops."""
    if not carte:
        return
    st.markdown(f"**{carte['titre']}**")
    if carte.get("note"):
        st.caption(carte["note"])
    if not carte["points"] and not carte["chemin"]:
        return
    components.html(to_leaflet_html(carte), height=440)
    if carte["trace"] == "approx":
        st.caption("Tracé approximatif : les arrêts sont reliés en pointillés "
                   "(pas de tracé GPS pour cette ligne dans les données).")
    elif carte["trace"] == "gps":
        st.caption(f"Tracé GPS : {len(carte['chemin'])} points.")


def show_chart(graphique: dict | None) -> None:
    """On-demand chart: single-series bars, real values from the evidence."""
    if not graphique:
        return
    st.markdown(f"**{graphique['titre']}**")
    st.altair_chart(bar_chart(graphique), use_container_width=True, theme="streamlit")
    notes = [n for n in (graphique.get("note"),
                         "Résultat tronqué : il existe d'autres valeurs." if graphique.get("tronque") else None) if n]
    if notes:
        st.caption(" ".join(notes))


def show_assistant(entry: dict, index: int) -> None:
    if entry.get("chemin") in ("echec", "quota"):
        st.warning(entry["content"])
    else:
        st.markdown(keep_line_breaks(entry["content"]))

    candidats = entry.get("candidats") or []
    is_last = index == len(st.session_state.history) - 1
    if candidats and is_last:
        cols = st.columns(len(candidats))
        for col, cand in zip(cols, candidats):
            if col.button(cand["nom"], key=f"cand-{index}-{cand['id']}"):
                st.session_state.pending = cand["nom"]
                st.rerun()

    show_map((entry.get("details") or {}).get("carte"))
    show_chart((entry.get("details") or {}).get("graphique"))
    show_how(entry.get("details") or {})


def ask(message: str) -> None:
    """Answer the user's last message, already shown in the conversation."""
    with st.chat_message("assistant"):
        with st.spinner("Je cherche dans les données SORETRAK…"):
            try:
                response = answer(message, st.session_state.thread_id)
                entry = {
                    "role": "assistant",
                    "content": response.texte,
                    "chemin": response.chemin,
                    "details": response.details,
                    "candidats": ((response.details.get("agent") or {}).get("resultat") or {})
                    .get("clarification", []) if response.chemin == "ambigu" else [],
                }
            except Exception:
                logger.exception("erreur inattendue")
                entry = {"role": "assistant", "content": respond.echec(), "chemin": "echec"}
    st.session_state.history.append(entry)
    st.rerun()




STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&family=Sora:wght@700&family=Source+Sans+3:wght@400;600;700&display=swap');

  :root {
    --vert: #43a047; --vert-fonce: #2e7d32; --vert-pale: #f1f5f1;
    --texte: #1f2937; --texte-2: #4b5563; --texte-3: #374151; --bordure: #e5e7eb;
  }

  /* Design font, without touching Streamlit's icons */
  .stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp label,
  .stApp button, .stApp textarea, .stApp input, .stApp td, .stApp th {
    font-family: 'Source Sans 3', sans-serif;
  }
  .stApp {  color: var(--texte); }
  .stApp a { color: var(--vert-fonce); }
  .stApp a:hover { color: #1b5e20; }

  /* Streamlit bar and menu hidden: the header is the design's own */
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
  [data-testid="stHeader"] { display: none; }

  /* Header: Tunisian Republic on the left, SORETRAK on the right, light shadow */
  .sa-header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 999;
    display: flex; justify-content: space-between; align-items: center; padding: 16px 32px;
    background: #fff; box-shadow: 0 2px 8px rgba(0, 0, 0, .08);
  }
  .sa-header .logo { height: 72px; max-width: 45%; object-fit: contain; }

  /* Central column of 1100 px, below the header */
  [data-testid="stMainBlockContainer"] {
    max-width: 1100px; padding: 136px 32px 40px;
  }

  /* Welcome screen */
.sa-welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: left;
    gap: 14px;
    margin: 72px 0 24px;
    max-width: 800px;
    padding-left : 50px ;
}
.sa-welcome h1 {
    font-family: 'Sora', sans-serif;
    font-size: 39px;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--vert-fonce);
    line-height: 1.25;
    animation: titleAppear 0.8s ease-out both;
}
  /* As soon as a message is shown, the welcome screen disappears (without waiting for the answer) */
  .stApp:has([data-testid="stChatMessage"]) .sa-welcome { display: none; }
  @keyframes titleAppear { from { opacity: 0; transform: translateY(12px); }
                           to   { opacity: 1; transform: none; } }
  /* Phone: the welcome screen fits on screen, below the header */
  @media (max-width: 640px) {
    .sa-welcome { margin-top: 8px; gap: 10px; }
    .sa-welcome h1 { font-size: 26px; }
    .sa-welcome p { font-size: 28px; }
  }
  .sa-welcome p { margin: 0; font-size: 25px; color: var(--texte-2); 
    width: 100%;
    text-align: left;
}
  


  

  /* "Nouvelle conversation": discreet link on the right */
  .st-key-newchat { margin-top: -8px; }
  .st-key-newchat [data-testid="stButton"] { display: flex; justify-content: flex-end; }
  .st-key-newchat button {
    border: none; background: none; color: var(--texte-2); font-size: 14px;
    padding: 2px 4px; min-height: 0; text-decoration: underline;
  }
  .st-key-newchat button:hover { color: var(--vert-fonce); background: none; }

  /* Bubbles without avatar: user on the right in green, assistant on the left in pale green */
  [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display: none; }
  [data-testid="stChatMessage"] {
    background: transparent; padding: 0; gap: 0; margin-bottom: 6px;
    width: 100%; display: flex;
  }
  [data-testid="stChatMessageContent"] {
    flex: 0 1 auto;
    max-width: 78%;
    width: fit-content;
    margin-right: auto;
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.5;
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    font-size: 22px !important;
    line-height: 1.5 !important;
}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    margin-left: auto; margin-right: 0;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--vert);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {
    color: #fff;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    flex: 0 1 78%; width: auto; margin: 0 auto 0 0 !important;   /* on the left: Streamlit used to center it */
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: var(--vert-pale); color: var(--texte);
  }
  /* Answer tables (timetables per line): green header, separated rows */
  [data-testid="stChatMessageContent"] table {
    border-collapse: collapse; width: 100%; margin: 8px 0; background: #fff;
    border-radius: 8px; overflow: hidden;
  }
  [data-testid="stChatMessageContent"] th,
  [data-testid="stChatMessageContent"] td {
    font-size: 20px; line-height: 1.45; padding: 8px 12px; text-align: left;
    border: none; border-bottom: 1px solid var(--bordure); vertical-align: top;
  }
  [data-testid="stChatMessageContent"] th { background: var(--vert); color: #fff; font-weight: 600; }
  [data-testid="stChatMessageContent"] td:first-child { font-weight: 600; color: var(--vert-fonce); white-space: nowrap; }
  [data-testid="stChatMessageContent"] tr:last-child td { border-bottom: none; }
  /* An answer with a map or chart takes the full width */
  [data-testid="stChatMessageContent"]:has(iframe, [data-testid="stVegaLiteChart"]) {
    flex: 1 1 auto; max-width: 100%;
  }

  /* Pill-shaped input with a caption below */
  [data-testid="stBottom"] > div { background: #fff; }
  [data-testid="stBottomBlockContainer"] { max-width: 1100px; padding: 18px 32px 15px; }
  [data-testid="stChatInput"] {
    border: 1px solid var(--bordure); border-radius: 999px; background: #fff;
    box-shadow: 0 2px 10px rgba(0, 0, 0, .08); padding-left: 12px;
  }
  [data-testid="stChatInput"] textarea::placeholder {
    font-size: 24px !important;
}
  [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"] [data-baseweb="textarea"],
  [data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: transparent !important; border: none !important; box-shadow: none !important;
  }
  [data-testid="stChatInput"] textarea {
    font-size: 24px !important;
}
 [data-testid="stChatInputSubmitButton"] {
    color: var(--texte-2);
    background: transparent;
    position: absolute !important;
    right: 10px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    margin: 0 !important;
}

[data-testid="stChatInputSubmitButton"]:hover {
    color: var(--vert-fonce);
    background: transparent;
}
  [data-testid="stBottomBlockContainer"]::after {
    content: "Propulsé par la technologie AI pour Kairouan";
    display: block; margin-top: 8px; text-align: right; font-size: 12px; color: #6b7280;
  }
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(
    '<header class="sa-header">'
    f'{logo_html("rbl_logo.png", "République tunisienne, Ministère du Transport")}'
    f'{logo_html("soretrak_logo.png", "SORETRAK")}'
    '</header>',
    unsafe_allow_html=True,
)

if not st.session_state.history:
    st.markdown(welcome_background_css(), unsafe_allow_html=True)   # illustrated background: welcome screen only
    st.markdown(
        """
        <section class="sa-welcome">
          <h1>Asslema, je suis ton assistant pour les lignes de transport en bus à Kairouan</h1>
          <p>Je peux te renseigner sur les lignes, les arrêts et les horaires théoriques</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

else:
    with st.container(key="newchat"):
        if st.button("Nouvelle conversation"):
            new_conversation()
            st.rerun()
    for i, entry in enumerate(st.session_state.history):
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant":
                show_assistant(entry, i)
            else:
                st.markdown(entry["content"])

typed = st.chat_input("Posez votre question ici…")
message = st.session_state.pending or typed
if message:
    st.session_state.pending = None
    st.session_state.history.append({"role": "user", "content": message})
    st.rerun()
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    ask(st.session_state.history[-1]["content"])
