"""Main graph: Router → place resolution → SQL Agent → Synthesizer | Respond.

One run per user message. The conversation state (history, context,
confirmed entities, pending clarification) is kept by the checkpointer,
one conversation per `thread_id`; its logic lives in memory.py.
"""
import logging
import operator
import time
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from soretrak import llm

from soretrak.components import respond, router, synthesizer
from soretrak.contracts import (AgentResult, AgentRun, Candidate, ResolvedPlace, Response,
                                TurnContext, TurnDetails)
from soretrak.presentation.carte import build_map
from soretrak.presentation.graphique import build_chart
from soretrak.conversation.memory import confirm, match_clarification, push_context, turn_context
from soretrak.places.facts import resolve_mentions
from soretrak.components.sql_agent import run_sql_agent

logger = logging.getLogger(__name__)

class ConversationState(TypedDict, total=False):
    # Conversation, persisted across turns
    messages: Annotated[list[dict], operator.add]
    entites_confirmees: list[dict]
    clarification_en_attente: list[dict]
    question_en_attente: dict | None   # question left pending by a Router clarification request
    contexte: list[TurnContext]   # factual summary of the last transport turns
    # Current turn, reset on every message
    message: str
    decision: dict | None
    lieux_resolus: list[ResolvedPlace]
    agent: AgentRun | None
    response: dict | None


def _llm_failure(error: llm.LLMError) -> dict:
    texte = respond.quota() if isinstance(error, llm.LLMRateLimitError) else respond.echec()
    chemin = "quota" if isinstance(error, llm.LLMRateLimitError) else "echec"
    return {"response": Response(texte=texte, chemin=chemin).model_dump()}


def router_node(state: ConversationState) -> dict:
    message = state["message"]
    history = state.get("messages", [])
    update: dict[str, Any] = {"messages": [{"role": "user", "content": message}],
                              "clarification_en_attente": [], "question_en_attente": None}
    en_attente = state.get("question_en_attente")

    pending = [Candidate(**c) for c in state.get("clarification_en_attente") or []]
    if pending:
        choice = match_clarification(message, pending)
        if choice:
            update["entites_confirmees"] = confirm(state.get("entites_confirmees") or [], choice)
            logger.info("clarification : %s confirmé", choice.nom)

    try:
        decision = router.route(message, history, state.get("contexte") or [], en_attente)
    except llm.LLMError as e:
        logger.warning("router : %s", e)
        return {**update, **_llm_failure(e)}

    update["decision"] = decision.model_dump()
    if decision.intent == "trop_vague":
        # The question stays pending: the next message will complete it.
        update["question_en_attente"] = {
            "question": f"{en_attente['question']} / {message}" if en_attente else message,
            "demande": decision.reponse,
        }
    if decision.intent != "transport":
        update["response"] = Response(texte=decision.reponse, chemin=decision.intent).model_dump()
    return update


def resolve_node(state: ConversationState) -> dict:
    """Resolve in code, without an LLM, the places picked up by the Router (pre-retrieval)."""
    entites = [Candidate(**c) for c in state.get("entites_confirmees") or []]
    return {"lieux_resolus": resolve_mentions(state["decision"].get("lieux") or [], entites)}


def sql_agent_node(state: ConversationState) -> dict:
    question = state["decision"]["question_autonome"]
    entites = [Candidate(**c) for c in state.get("entites_confirmees") or []]
    final = run_sql_agent(question, entites, lieux_resolus=state.get("lieux_resolus") or [])
    result: AgentResult = final["resultat"]
    run: AgentRun = {
        "resultat": result.model_dump(),
        "requetes": final.get("requetes", {}),
        "trace": final.get("messages", [])[2:],
        "nb_appels": final.get("nb_appels", 0),
    }
    update: dict[str, Any] = {"agent": run}
    if result.statut == "ambigu":
        update["clarification_en_attente"] = [c.model_dump() for c in result.clarification]
    entry = turn_context(question, state.get("lieux_resolus") or [], run)
    update["contexte"] = push_context(state.get("contexte") or [], entry)
    return update


def synthesizer_node(state: ConversationState) -> dict:
    agent = state["agent"]
    result = AgentResult(**agent["resultat"])
    try:
        texte = synthesizer.synthesize(state["decision"]["question_autonome"], result,
                                       agent["requetes"],
                                       graphique=bool(state["decision"].get("graphique")))
    except llm.LLMError as e:
        logger.warning("synthesizer : %s", e)
        return _llm_failure(e)
    return {"response": Response(texte=texte, chemin=result.statut).model_dump()}


def respond_node(state: ConversationState) -> dict:
    result = AgentResult(**state["agent"]["resultat"])
    if result.quota:
        return _llm_failure(llm.LLMRateLimitError(result.note))
    return {"response": Response(texte=respond.for_agent_result(result, state.get("lieux_resolus")),
                                 chemin=result.statut).model_dump()}


def finalize_node(state: ConversationState) -> dict:
    """Append the answer to the history."""
    return {"messages": [{"role": "assistant", "content": state["response"]["texte"]}]}


def _after_router(state: ConversationState) -> str:
    return "finalize" if state.get("response") else "resolve"


def _after_agent(state: ConversationState) -> str:
    statut = state["agent"]["resultat"]["statut"]
    return "synthesizer" if statut in ("repondu", "aucune_donnee") else "respond"


def build_graph(checkpointer=None):
    graph = StateGraph(ConversationState)
    graph.add_node("router", router_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("respond", respond_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _after_router, ["resolve", "finalize"])
    graph.add_edge("resolve", "sql_agent")
    graph.add_conditional_edges("sql_agent", _after_agent, ["synthesizer", "respond"])
    graph.add_edge("synthesizer", "finalize")
    graph.add_edge("respond", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def answer(message: str, thread_id: str, graph=None) -> Response:
    """Entry point: a user message → the assistant's answer."""
    graph = graph or get_graph()
    run_id = uuid.uuid4().hex[:8]
    start = time.monotonic()
    state = graph.invoke(
        {"message": message, "decision": None, "lieux_resolus": [],
         "agent": None, "response": None},
        {"configurable": {"thread_id": thread_id}},
    )
    response = Response(**state["response"])
    decision = state.get("decision") or {}
    details: TurnDetails = {
        "decision": state.get("decision"),
        "agent": state.get("agent"),
        "entites_confirmees": state.get("entites_confirmees") or [],
        "lieux": state.get("lieux_resolus") or [],
        # Map on demand only, built by the code from the resolved places.
        "carte": build_map(state.get("lieux_resolus") or []) if decision.get("carte") else None,
        # Chart on demand only, from the real rows of the evidence.
        "graphique": (build_chart(decision.get("question_autonome") or message, state.get("agent"))
                      if decision.get("graphique") else None),
    }
    response.details = dict(details)
    agent = state.get("agent") or {}
    logger.info(
        "tour run=%s thread=%s intent=%s chemin=%s appels=%s duree=%.1fs",
        run_id, thread_id, (state.get("decision") or {}).get("intent"),
        response.chemin, agent.get("nb_appels", 0), time.monotonic() - start,
    )
    return response
