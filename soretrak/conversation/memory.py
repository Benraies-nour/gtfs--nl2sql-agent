"""
The message history and persistence are handled by the graph (checkpointer).
"""
from rapidfuzz import fuzz

from soretrak.contracts import AgentRun, Candidate, ResolvedPlace, TurnContext
from soretrak.places.normalize import normalize

CLARIFICATION_MATCH_SCORE = 85
CONTEXT_TURNS = 3       # transport turns kept in the context
CONTEXT_MAX_VALUES = 12


def turn_context(question: str, lieux_resolus: list[ResolvedPlace], agent: AgentRun) -> TurnContext:
    """Factual summary of a turn, built by the code: places, lines and stops involved."""
    lieux = [
        {"type": c["type"], "id": c["id"], "nom": c["nom"]}
        for lieu in lieux_resolus if lieu["statut"] == "trouve"
        for c in lieu["candidats"][:1]
    ]
    lignes, arrets = [], []
    for qid in agent["resultat"]["preuves"]:
        for row in agent["requetes"].get(qid, {}).get("rows", []):
            if row.get("route_id") is not None:
                lignes.append(str(row["route_id"]))
            if row.get("lignes"):                     # q0 facts: "3, 21, 22"
                lignes.extend(v.strip() for v in str(row["lignes"]).split(","))
            if row.get("stop_name"):
                arrets.append(row["stop_name"])
    return {
        "question": question,
        "lieux": lieux,
        "lignes": list(dict.fromkeys(lignes))[:CONTEXT_MAX_VALUES],
        "arrets": list(dict.fromkeys(arrets))[:CONTEXT_MAX_VALUES],
        "statut": agent["resultat"]["statut"],
    }


def push_context(contexte: list[TurnContext], entry: TurnContext) -> list[TurnContext]:
    """Append a turn summary and keep only the most recent ones."""
    return (contexte + [entry])[-CONTEXT_TURNS:]


def match_clarification(message: str, candidats: list[Candidate]) -> Candidate | None:
    """Match the user's reply against the proposed candidates.

    Accepts the name ("Bab Jedid", even misspelled) or the rank ("1", "le 2").
    """
    text = normalize(message)
    words = text.split()
    if len(words) <= 2 and words and words[-1].isdigit():
        rank = int(words[-1])
        if 1 <= rank <= len(candidats):
            return candidats[rank - 1]
    scored = sorted(((fuzz.ratio(text, normalize(c.nom)), c) for c in candidats),
                    key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < CLARIFICATION_MATCH_SCORE:
        return None
    if len(scored) > 1 and scored[1][0] >= CLARIFICATION_MATCH_SCORE:
        return None
    return scored[0][1]


def confirm(entites: list[dict], choice: Candidate) -> list[dict]:
    """Add the chosen entity to the confirmed entities (no duplicates)."""
    others = [c for c in entites if (c["type"], c["id"]) != (choice.type, choice.id)]
    return others + [choice.model_dump()]
