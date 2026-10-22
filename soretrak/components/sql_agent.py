"""SQL Agent: ReAct subgraph agent ⇄ executor.

Receives the Router's standalone question and returns a status and evidence
(AgentResult) with the query registry. The code sets the limits: tool-call
budget, number of turns, `echec` status when something goes wrong.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph

from soretrak import llm
from soretrak.contracts import AgentResult, Candidate, QueryRecord, ResolvedPlace
from soretrak.places.facts import facts_query, format_lieux
from soretrak.prompts import knowledge, render
from soretrak.components.tools import (COUNTED_TOOLS, TOOL_SCHEMAS, run_finish, run_resolve_place,
                            run_run_sql)

logger = logging.getLogger(__name__)

TOOL_BUDGET = 4     # places are resolved before the agent (pre-retrieval)
MAX_TURNS = 10      # total LLM turns, guard against a loop on an invalid finish
TZ = ZoneInfo("Africa/Tunis")


class AgentState(TypedDict, total=False):
    question: str
    entites_confirmees: list[Candidate]
    db_path: str | None
    messages: list[dict]
    requetes: dict[str, QueryRecord]
    nb_appels: int
    nb_tours: int
    resultat: AgentResult | None


def build_system_prompt(entites_confirmees: list[Candidate], now: datetime | None = None) -> str:
    now = now or datetime.now(TZ)
    entites = ", ".join(f"{c.nom} ({c.type} {c.id})" for c in entites_confirmees) or "aucune"
    return render(
        "sql_agent.md",
        maintenant=now.strftime("%H:%M:%S"),
        entites=entites,
        schema=knowledge("schema.md"),
        examples=knowledge("examples.yaml"),
        perimetre=knowledge("perimetre.md"),
    )


def build_user_message(question: str, lieux_resolus: list[ResolvedPlace]) -> str:
    """The question, followed by what the code already knows about its places (evidence q0).

    The facts sit next to the question, not deep in the system prompt:
    far from the question, the agent ignored them and recomputed them in SQL.
    """
    if not lieux_resolus:
        return question
    return (f"{question}\n\nDéjà connu (lieux résolus par le code, preuve q0 : "
            f"ne les recherche pas en SQL) :\n{format_lieux(lieux_resolus)}")


def _fail(note: str) -> dict:
    return {"resultat": AgentResult(statut="echec", note=note)}


def agent_node(state: AgentState) -> dict:
    if state.get("nb_tours", 0) >= MAX_TURNS:
        return _fail("nombre maximal de tours atteint")
    try:
        message = llm.with_tools(state["messages"], TOOL_SCHEMAS)
    except llm.LLMRateLimitError as e:
        logger.warning("sql_agent : limite de débit : %s", e)
        return {"resultat": AgentResult(statut="echec", note=f"erreur LLM : {e}", quota=True)}
    except llm.LLMError as e:
        logger.warning("sql_agent : erreur LLM : %s", e)
        return _fail(f"erreur LLM : {e}")

    assistant = {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ],
    }
    return {"messages": state["messages"] + [assistant], "nb_tours": state.get("nb_tours", 0) + 1}


def executor_node(state: AgentState) -> dict:
    messages = list(state["messages"])
    requetes = dict(state.get("requetes", {}))
    nb_appels = state.get("nb_appels", 0)

    for call in messages[-1]["tool_calls"]:
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = None

        if name in COUNTED_TOOLS and nb_appels >= TOOL_BUDGET:
            return {"messages": messages, "requetes": requetes, "nb_appels": nb_appels,
                    **_fail("budget d'appels d'outils épuisé")}

        if not isinstance(args, dict):
            content = "Arguments invalides : un objet JSON est attendu."
        elif name == "resolve_place":
            nb_appels += 1
            content = run_resolve_place(args, state.get("entites_confirmees", []),
                                        state.get("db_path"))
        elif name == "run_sql":
            nb_appels += 1
            content = run_run_sql(args, requetes, state.get("db_path"))
        elif name == "finish":
            outcome = run_finish(args, requetes)
            if isinstance(outcome, AgentResult) and outcome.statut == "impossible" \
                    and nb_appels >= TOOL_BUDGET:
                # Concluding "impossible" for lack of remaining calls is a failure, decided by the code.
                outcome = AgentResult(statut="echec",
                                      note=f"budget d'appels d'outils épuisé ; note de l'agent : {outcome.note}")
            if isinstance(outcome, AgentResult):
                return {"messages": messages, "requetes": requetes,
                        "nb_appels": nb_appels, "resultat": outcome}
            content = outcome
        else:
            content = f"Outil inconnu : {name}."

        if name in COUNTED_TOOLS and nb_appels == TOOL_BUDGET:
            content += "\n[Budget épuisé : c'était ton dernier appel. Appelle finish maintenant.]"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    return {"messages": messages, "requetes": requetes, "nb_appels": nb_appels}


def _after(state: AgentState) -> str:
    return END if state.get("resultat") else "next"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("executor", executor_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _after, {"next": "executor", END: END})
    graph.add_conditional_edges("executor", _after, {"next": "agent", END: END})
    return graph.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_sql_agent(
    question: str,
    entites_confirmees: list[Candidate] | None = None,
    now: datetime | None = None,
    db_path: str | Path | None = None,
    lieux_resolus: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the agent. Returns the final state: resultat, requetes, messages, nb_appels."""
    entites = entites_confirmees or []
    lieux_resolus = lieux_resolus or []
    q0 = facts_query(lieux_resolus)
    state: AgentState = {
        "question": question,
        "entites_confirmees": entites,
        "db_path": str(db_path) if db_path else None,
        "messages": [
            {"role": "system", "content": build_system_prompt(entites, now)},
            {"role": "user", "content": build_user_message(question, lieux_resolus)},
        ],
        "requetes": {"q0": q0} if q0 else {},
        "nb_appels": 0,
        "nb_tours": 0,
        "resultat": None,
    }
    final = _get_graph().invoke(state, {"recursion_limit": 2 * MAX_TURNS + 5})
    logger.info("sql_agent statut=%s appels=%s tours=%s", final["resultat"].statut,
                final.get("nb_appels"), final.get("nb_tours"))
    return final
