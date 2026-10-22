"""Respond: a fixed message for every case without a data answer.
"""
from soretrak.contracts import AgentResult, ResolvedPlace

MESSAGE_ECHEC = (
    "Désolé, je n'ai pas réussi à trouver la réponse cette fois-ci. "
    "Pouvez-vous reformuler votre question ?"
)

MESSAGE_QUOTA = (
    "Le service est très sollicité en ce moment. "
    "Merci de réessayer dans une minute."
)


def ambigu(result: AgentResult) -> str:
    names = [c.nom for c in result.clarification]
    if not names:
        return "Pouvez-vous préciser l'arrêt ou la ligne dont vous parlez ?"
    if len(names) == 1:
        return f"Parlez-vous de {names[0]} ?"
    types = {c.type for c in result.clarification}
    what = {frozenset({"arret"}): "arrêts", frozenset({"ligne"}): "lignes"}.get(
        frozenset(types), "lieux")
    listed = ", ".join(names[:-1]) + f" ou {names[-1]}"
    return f"Plusieurs {what} correspondent. Parlez-vous de {listed} ?"


CONTACT = "SORETRAK (www.soretrak.com.tn, +216 77 226 321)"


def impossible(result: AgentResult, lieux_resolus: list[ResolvedPlace] | None = None) -> str:
    """Never the agent's note (internal memo, sometimes technical): only what the
    code knows itself, i.e. the places that were not found."""
    missing = [l["texte"] for l in lieux_resolus or [] if l["statut"] == "introuvable"]
    if missing:
        names = " et ".join(f"« {m} »" for m in missing)
        text = f"Je n'ai pas trouvé {names} parmi les arrêts et les lignes SORETRAK."
    else:
        text = "Je n'ai pas trouvé cette information dans les données SORETRAK."
    return f"{text} Pour aller plus loin, vous pouvez contacter {CONTACT}."


def echec() -> str:
    return MESSAGE_ECHEC


def quota() -> str:
    return MESSAGE_QUOTA


def for_agent_result(result: AgentResult, lieux_resolus: list[ResolvedPlace] | None = None) -> str:
    """Fixed message for the statuses that do not go through the Synthesizer."""
    if result.statut == "ambigu":
        return ambigu(result)
    if result.statut == "impossible":
        return impossible(result, lieux_resolus)
    return echec()
