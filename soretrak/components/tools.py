"""The 3 SQL Agent tools: their schema sent to the LLM and their execution.

The LLM picks the tool and its arguments; the code runs it. A tool error
is returned to the agent as a result, so that it can correct itself.
"""
import json
import re
from pathlib import Path

from pydantic import ValidationError

from soretrak.config import DB_PATH
from soretrak.contracts import AgentResult, Candidate, QueryRecord
from soretrak.sql.executor import run_query
from soretrak.places.resolver import resolve_place

COUNTED_TOOLS = {"resolve_place", "run_sql"}   # finish does not count against the budget
# Statuses the agent may choose; `echec` is reserved to the code.
AGENT_STATUSES = ("repondu", "aucune_donnee", "ambigu", "impossible")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_place",
            "description": (
                "Trouve l'arrêt ou la ligne correspondant à un texte de l'utilisateur "
                "(« bab jdid », « bus 21 », « Sousse »). Renvoie un statut "
                "(trouve / ambigu / introuvable) et jusqu'à 5 candidats avec leur id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texte": {"type": "string", "description": "Le nom tel que l'utilisateur l'a écrit"},
                    "type": {
                        "type": "string",
                        "enum": ["arret", "ligne", "ville"],
                        "description": "Optionnel. « ligne » pour chercher dans les noms longs des lignes.",
                    },
                },
                "required": ["texte"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Exécute une seule requête SELECT ou WITH (lecture seule). Renvoie un id "
                "(q1, q2…), les colonnes, au plus 50 lignes, le nombre de lignes et "
                "si le résultat est tronqué ; ou un message d'erreur."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    # The "think" step of ReAct: tool_choice="required" prevents any free text.
                    "raison": {"type": "string", "description": (
                        "Une phrase : ce que cette requête doit établir et pourquoi, "
                        "compte tenu des résultats précédents.")},
                    "sql": {"type": "string"},
                },
                "required": ["raison", "sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Termine la recherche avec un statut et les ids des requêtes qui servent de preuves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statut": {
                        "type": "string",
                        "enum": list(AGENT_STATUSES),
                    },
                    "preuves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ids des requêtes retenues (q1, q2…). Obligatoire pour repondu et aucune_donnee.",
                    },
                    "note": {"type": "string", "description": "Une note courte (partie non répondable, raison…)"},
                    "clarification": {
                        "type": "array",
                        "description": "Pour ambigu : les candidats à proposer à l'utilisateur.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["arret", "ligne"]},
                                "id": {"type": "string"},
                                "nom": {"type": "string"},
                            },
                            "required": ["type", "id", "nom"],
                        },
                    },
                },
                "required": ["statut"],
            },
        },
    },
]


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def run_resolve_place(args: dict, entites_confirmees: list[Candidate],
                      db_path: str | Path | None = None) -> str:
    texte = args.get("texte")
    if not isinstance(texte, str) or not texte.strip():
        return _json({"erreur": "Argument « texte » manquant."})
    type_ = args.get("type") if args.get("type") in ("arret", "ligne", "ville") else None
    result = resolve_place(texte, type=type_, entites_confirmees=entites_confirmees,
                           db_path=db_path or DB_PATH)
    return result.model_dump_json()


_TEXT_TIME_ARITHMETIC = re.compile(r"\b(?:departure|arrival)_time\b\s*-(?!-)|-\s*\w*\.?(?:departure|arrival)_time\b",
                                   re.IGNORECASE)
_DURATION_COLUMN = re.compile(r"duree|durée|minute|temps", re.IGNORECASE)


def warnings_for(sql: str, result) -> list[str]:
    """Deterministic signals on a suspicious result, read by the agent to correct itself.

    The code fixes nothing: it flags, and the agent decides what to do.
    """
    warnings = []
    if _TEXT_TIME_ARITHMETIC.search(re.sub(r"strftime\([^)]*\)", "", sql)):
        warnings.append("Soustraction d'heures en texte : '06:46:00' - '06:30:00' vaut 0 (seules les "
                        "heures sont soustraites). Utilise strftime('%s', …) pour une durée.")
    durations = [c for c in result.columns if _DURATION_COLUMN.search(c)]
    if any(isinstance(row.get(c), (int, float)) and row[c] <= 0 for row in result.rows for c in durations):
        warnings.append("Durée nulle ou négative : calcul sur des heures en texte, ou arrêts pris "
                        "dans le mauvais ordre (A doit précéder B : a.stop_sequence < b.stop_sequence) ?")
    # No signal on an empty result: measured, it made the agent doubt a correct
    # empty result ("no direct line") and doubled its calls without changing the conclusion.
    return warnings


def run_run_sql(args: dict, requetes: dict[str, QueryRecord], db_path: str | Path | None = None) -> str:
    """Run the query and record it in `requetes` (modified in place)."""
    sql = args.get("sql")
    if not isinstance(sql, str):
        return _json({"erreur": "Argument « sql » manquant."})
    result = run_query(sql, db_path or DB_PATH)
    qid = f"q{sum(1 for k in requetes if k != 'q0') + 1}"   # q0: facts about the places
    requetes[qid] = {"sql": sql, **result.model_dump()}
    if not result.ok:
        return _json({"id": qid, "erreur": result.error})
    observation = {
        "id": qid,
        "colonnes": result.columns,
        "lignes": result.rows,
        "nb_lignes": result.row_count,
        "tronque": result.truncated,
    }
    warnings = warnings_for(sql, result)
    if warnings:
        observation["attention"] = warnings
    return _json(observation)


def run_finish(args: dict, requetes: dict[str, QueryRecord]) -> AgentResult | str:
    """Return the AgentResult if the call is valid, otherwise an error message for the agent."""
    preuves = args.get("preuves") or []
    unknown = [p for p in preuves if p not in requetes]
    if unknown:
        return f"Preuves inexistantes : {', '.join(unknown)}. Ids disponibles : {', '.join(requetes) or 'aucun'}."
    failed = [p for p in preuves if requetes[p].get("error")]
    if failed:
        return f"Ces requêtes ont échoué et ne peuvent pas servir de preuve : {', '.join(failed)}."

    statut = args.get("statut")
    if statut not in AGENT_STATUSES:
        return f"Statut invalide : {statut!r}. Choisir parmi {', '.join(AGENT_STATUSES)}."
    if statut in ("repondu", "aucune_donnee") and not preuves:
        return f"Le statut {statut} demande au moins une preuve (id de requête)."
    if statut == "ambigu" and not args.get("clarification"):
        return "Le statut ambigu demande la liste des candidats dans « clarification »."

    try:
        return AgentResult(
            statut=statut,
            preuves=preuves,
            note=args.get("note") or None,
            clarification=[
                Candidate(score=0, raison="", **c) for c in args.get("clarification") or []
            ],
        )
    except (ValidationError, TypeError) as e:
        return f"Appel à finish invalide : {e}"
