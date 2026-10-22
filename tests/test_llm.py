"""LLM gateway with a fake client (no network call)."""
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import ValidationError

from soretrak import config, llm
from soretrak.contracts import RouterDecision


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _api_error(cls, status):
    request = httpx.Request("POST", "https://example.invalid")
    return cls("erreur", response=httpx.Response(status, request=request), body=None)


class FakeClient:
    """Replay a list of responses or exceptions, and record the calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(config, "LLM_RETRY_DELAY_SECONDS", 0)

    def install(*outcomes):
        client = FakeClient(*outcomes)
        llm.set_client(client)
        return client

    yield install
    llm.set_client(None)


# --- contracts ---------------------------------------------------------------

def test_router_decision_transport_needs_question():
    with pytest.raises(ValidationError):
        RouterDecision(intent="transport")
    RouterDecision(intent="transport", question_autonome="Horaires de la ligne 22")


def test_router_decision_other_intents_need_reponse():
    with pytest.raises(ValidationError):
        RouterDecision(intent="conversation")
    RouterDecision(intent="hors_sujet", reponse="Je réponds sur les bus SORETRAK.")


# --- structured() -----------------------------------------------------------

def test_structured_valid(fake):
    client = fake(_response('{"intent": "conversation", "reponse": "Bonjour !"}'))
    decision = llm.structured("système", "asslema", RouterDecision)
    assert decision.intent == "conversation"
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["temperature"] == 0


def test_structured_retries_once_on_invalid_output(fake):
    client = fake(
        _response('{"intent": "transport"}'),  # missing question_autonome
        _response('{"intent": "transport", "question_autonome": "Lignes à Bab Jedid"}'),
    )
    decision = llm.structured("système", "lignes bab jdid", RouterDecision)
    assert decision.question_autonome == "Lignes à Bab Jedid"
    assert len(client.calls) == 2
    assert "Sortie invalide" in client.calls[1]["messages"][-1]["content"]


def test_structured_gives_up_after_second_invalid_output(fake):
    fake(_response("pas du json"), _response("{}"))
    with pytest.raises(llm.LLMError):
        llm.structured("système", "x", RouterDecision)


# --- retries and error translation -------------------------------------------

def test_retry_on_429_then_success(fake):
    client = fake(
        _api_error(openai.RateLimitError, 429),
        _response('{"intent": "conversation", "reponse": "ok"}'),
    )
    llm.structured("s", "u", RouterDecision)
    assert len(client.calls) == 2


def test_rate_limit_after_all_retries(fake):
    fake(*[_api_error(openai.RateLimitError, 429)] * (config.LLM_RETRIES + 1))
    with pytest.raises(llm.LLMRateLimitError):
        llm.structured("s", "u", RouterDecision)


def test_retry_on_5xx(fake):
    client = fake(
        _api_error(openai.InternalServerError, 503),
        _response('{"intent": "conversation", "reponse": "ok"}'),
    )
    llm.structured("s", "u", RouterDecision)
    assert len(client.calls) == 2


def test_no_retry_on_400(fake):
    client = fake(_api_error(openai.BadRequestError, 400))
    with pytest.raises(llm.LLMError) as info:
        llm.structured("s", "u", RouterDecision)
    assert not isinstance(info.value, llm.LLMRateLimitError)
    assert len(client.calls) == 1


# --- with_tools() -----------------------------------------------------------

def test_with_tools_requires_a_tool(fake):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="run_sql", arguments="{}"))
    client = fake(_response(tool_calls=[call]))
    message = llm.with_tools([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert message.tool_calls[0].function.name == "run_sql"
    assert client.calls[0]["tool_choice"] == "required"
    assert client.calls[0]["parallel_tool_calls"] is False


def test_with_tools_without_tool_call_is_an_error(fake):
    fake(_response(content="je réponds sans outil"))
    with pytest.raises(llm.LLMError):
        llm.with_tools([{"role": "user", "content": "x"}], tools=[{"type": "function"}])


def test_waits_for_delay_requested_by_provider(fake, monkeypatch):
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    error = _api_error(openai.RateLimitError, 429)
    error.response.headers["retry-after"] = "7"
    fake(error, _response('{"intent": "conversation", "reponse": "ok"}'))
    llm.structured("s", "u", RouterDecision)
    assert waits == [7.0]


def test_reads_delay_from_groq_message(fake, monkeypatch):
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    request = httpx.Request("POST", "https://example.invalid")
    error = openai.RateLimitError("Please try again in 22.5s.",
                                  response=httpx.Response(429, request=request), body=None)
    fake(error, _response('{"intent": "conversation", "reponse": "ok"}'))
    llm.structured("s", "u", RouterDecision)
    assert waits == [22.5]


def test_gives_up_when_requested_delay_is_too_long(fake, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: pytest.fail("ne doit pas attendre"))
    error = _api_error(openai.RateLimitError, 429)
    error.response.headers["retry-after"] = "120"
    client = fake(error)
    with pytest.raises(llm.LLMRateLimitError):
        llm.structured("s", "u", RouterDecision)
    assert len(client.calls) == 1


def test_retry_on_groq_output_parse_failed(fake):
    request = httpx.Request("POST", "https://example.invalid")
    error = openai.BadRequestError(
        "Parsing failed", response=httpx.Response(400, request=request),
        body={"code": "output_parse_failed", "message": "Parsing failed"},
    )
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="finish", arguments="{}"))
    client = fake(error, _response(tool_calls=[call]))
    llm.with_tools([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert len(client.calls) == 2



def test_retry_on_groq_json_validate_failed(fake):
    request = httpx.Request("POST", "https://example.invalid")
    error = openai.BadRequestError(
        "Failed to validate JSON", response=httpx.Response(400, request=request),
        body={"code": "json_validate_failed", "message": "Failed to validate JSON"},
    )
    client = fake(error, _response('{"intent": "conversation", "reponse": "ok"}'))
    llm.structured("s", "u", RouterDecision)
    assert len(client.calls) == 2
