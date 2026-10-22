"""Pre-retrieval: the places of the question, resolved by the code before the agent.


"""
from pathlib import Path

from soretrak.config import DB_PATH
from soretrak.contracts import Candidate, PlaceFacts, QueryRecord, ResolvedPlace
from soretrak.places.resolver import resolve_place
from soretrak.sql.executor import run_query


def place_facts(candidat: Candidate, db_path: str | Path | None = None) -> PlaceFacts:
    """Useful facts about a resolved place, so that the agent does not have to look them up.

    Line: does it have timetables, how many stops, which directions.
    Stop: the lines that serve it.
    """
    db = db_path or DB_PATH
    facts = {"type": candidat.type, "id": candidat.id, "nom": candidat.nom}
    if candidat.type == "ligne":
        result = run_query(
            "SELECT COUNT(DISTINCT stop_id) AS nb_arrets, "
            "GROUP_CONCAT(DISTINCT direction_id) AS sens "
            "FROM v_route_stops WHERE route_id = ?",
            db, params=(candidat.id,),
        )
        row = result.rows[0] if result.ok and result.rows else {}
        facts["nb_arrets"] = row.get("nb_arrets") or 0
        facts["a_horaires"] = facts["nb_arrets"] > 0
        facts["sens"] = sorted(int(s) for s in (row.get("sens") or "").split(",") if s != "")
    else:
        result = run_query(
            "SELECT DISTINCT route_id FROM v_route_stops "
            "WHERE stop_id = ? ORDER BY CAST(route_id AS INTEGER)",
            db, params=(candidat.id,),
        )
        facts["lignes"] = [r["route_id"] for r in result.rows] if result.ok else []
    return facts


def resolve_mentions(lieux: list[str], entites_confirmees: list[Candidate] | None = None,
                     db_path: str | Path | None = None) -> list[ResolvedPlace]:
    """Resolve the places mentioned by the user, before the agent, with their facts."""
    resolved = []
    for texte in dict.fromkeys(t.strip() for t in lieux if t and t.strip()):
        result = resolve_place(texte, entites_confirmees=entites_confirmees,
                               db_path=db_path or DB_PATH)
        resolved.append({
            "texte": texte,
            "statut": result.statut,
            "candidats": [{**place_facts(c, db_path), "score": c.score}
                          for c in result.candidats],
        })
    return resolved


def format_lieux(lieux_resolus: list[ResolvedPlace]) -> str:
    """The places of the question, resolved by the code, in plain text for the agent."""
    if not lieux_resolus:
        return "aucun lieu relevé dans la question (utilise resolve_place si besoin)"
    lines = []
    for lieu in lieux_resolus:
        if lieu["statut"] == "introuvable":
            lines.append(f"- « {lieu['texte']} » → introuvable")
            continue
        if lieu["statut"] == "ambigu":
            lines.append(f"- « {lieu['texte']} » → AMBIGU. Choisis UN seul candidat si la question "
                         "le désigne clairement ; sinon finish(ambigu) avec ces candidats, sans SQL. "
                         "N'interroge jamais plusieurs candidats ensemble :")
        else:
            lines.append(f"- « {lieu['texte']} » → {lieu['statut']} :")
        for c in lieu["candidats"]:
            if c["type"] == "ligne":
                horaires = "oui" if c["a_horaires"] else "non"
                sens = ", ".join(map(str, c["sens"])) or "aucun"
                lines.append(f"    - ligne {c['id']} « {c['nom']} » (route_id '{c['id']}') ; "
                             f"horaires : {horaires} ; {c['nb_arrets']} arrêts ; sens : {sens}")
            else:
                lignes = ", ".join(c["lignes"]) or "aucune"
                lines.append(f"    - arrêt « {c['nom']} » (stop_id '{c['id']}') ; "
                             f"lignes qui y passent : {lignes}")
    return "\n".join(lines)


def facts_query(lieux_resolus: list[ResolvedPlace]) -> QueryRecord | None:
    """The places' facts, recorded as evidence q0 (citable in finish)."""
    rows = []
    for lieu in lieux_resolus:
        for c in lieu["candidats"] or [{}]:
            row = {"texte_utilisateur": lieu["texte"], "statut": lieu["statut"]}
            row.update({k: ", ".join(v) if isinstance(v, list) and v and isinstance(v[0], str)
                        else v for k, v in c.items() if k != "score"})
            rows.append(row)
    if not rows:
        return None
    columns = list(dict.fromkeys(k for row in rows for k in row))
    return {"sql": "(lieux résolus par le code avant l'agent)", "columns": columns,
            "rows": rows, "row_count": len(rows), "truncated": False, "error": None}
