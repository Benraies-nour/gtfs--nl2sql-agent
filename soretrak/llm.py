"""LLM gateway: a single OpenAI-compatible client (Groq or OpenAI).
"""
import json
import logging
import re
import time
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from soretrak import config

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class LLMError(Exception):
    """LLM call failure (network, API, or unusable output)."""


class LLMRateLimitError(LLMError):
    """Rate limit or quota reached, even after retries."""


_client: openai.OpenAI | None = None

# Cumulative usage since startup (read by evals/run.py).
USAGE = {"appels": 0, "tokens_in": 0, "tokens_out": 0}


def get_client() -> openai.OpenAI:
    """Create the client on first use, never at import time."""
    global _client
    if _client is None:
        provider = config.PROVIDERS[config.LLM_PROVIDER]
        if not provider["api_key"]:
            raise LLMError(f"Clé API absente pour le fournisseur {config.LLM_PROVIDER}")
        _client = openai.OpenAI(
            api_key=provider["api_key"],
            base_url=provider["base_url"],
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,  # retries are handled here
        )
    return _client


def set_client(client) -> None:
    """Replace the client (tests: fake client without network access)."""
    global _client
    _client = client


def _wait_seconds(error: Exception, attempt: int) -> float:
    """Return the retry delay requested by the provider, or a short fallback delay."""
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error))
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2))
    return config.LLM_RETRY_DELAY_SECONDS * attempt


def _call(model: str, **kwargs) -> Any:
    """Call chat.completions with retries for 429 and 5xx errors."""
    attempts = config.LLM_RETRIES + 1
    for attempt in range(1, attempts + 1):
        start = time.monotonic()
        try:
            response = get_client().chat.completions.create(model=model, **kwargs)
        except openai.RateLimitError as e:
            error = LLMRateLimitError(f"Limite de débit atteinte ({model})")
            retry = True
            cause = e
        except openai.InternalServerError as e:
            error, retry, cause = LLMError(f"Erreur serveur du fournisseur ({model})"), True, e
        except openai.APIConnectionError as e:
            error, retry, cause = LLMError(f"Connexion au fournisseur impossible ({model})"), True, e
        except openai.BadRequestError as e:
            # Groq / gpt-oss: malformed transient generation, retry it.
            retry = getattr(e, "code", None) in (
                "output_parse_failed", "tool_use_failed", "json_validate_failed")
            error, cause = LLMError(f"Appel LLM refusé ({model}) : {e}"), e
        except openai.OpenAIError as e:
            error, retry, cause = LLMError(f"Appel LLM refusé ({model}) : {e}"), False, e
        else:
            usage = getattr(response, "usage", None)
            USAGE["appels"] += 1
            USAGE["tokens_in"] += getattr(usage, "prompt_tokens", 0) or 0
            USAGE["tokens_out"] += getattr(usage, "completion_tokens", 0) or 0
            logger.info(
                "llm model=%s latence=%.2fs tokens_in=%s tokens_out=%s",
                model, time.monotonic() - start,
                getattr(usage, "prompt_tokens", "?"), getattr(usage, "completion_tokens", "?"),
            )
            return response

        logger.warning("llm model=%s essai %d/%d échoué : %s", model, attempt, attempts, cause)
        wait = _wait_seconds(cause, attempt)
        if not retry or attempt == attempts or wait > config.LLM_MAX_WAIT_SECONDS:
            raise error from cause
        logger.info("llm nouvel essai dans %.1fs", wait)
        time.sleep(wait)


def _describe(prop: dict) -> Any:
    """Create a readable example value for a JSON schema field."""
    options = prop.get("anyOf")
    if options:
        parts = [_describe(o) for o in options if o.get("type") != "null"]
        nullable = any(o.get("type") == "null" for o in options)
        value = parts[0] if len(parts) == 1 else " | ".join(map(str, parts))
        return f"{value} ou null" if nullable and isinstance(value, str) else value
    if "enum" in prop:
        return "|".join(prop["enum"])
    if prop.get("type") == "array":
        return [_describe(prop.get("items", {}))]
    return {"string": "texte", "integer": "nombre", "number": "nombre",
            "boolean": "true|false"}.get(prop.get("type"), "valeur")


def json_shape(schema: type[BaseModel]) -> str:
    """Build a simplified description of the expected object instead of the raw JSON schema.

    Some models (gpt-4o-mini) copy the JSON schema instead of filling it.
    """
    props = schema.model_json_schema().get("properties", {})
    return json.dumps({name: _describe(prop) for name, prop in props.items()},
                      ensure_ascii=False, indent=1)


def structured(system: str, user: str, schema: type[M], model: str | None = None) -> M:
    """Request JSON output matching `schema`, with one retry if invalid."""
    model = model or config.MODEL_ROUTER
    system = (
        f"{system}\n\nRéponds uniquement par un objet JSON de cette forme "
        f"(remplis les valeurs, ne renvoie pas cette description) :\n{json_shape(schema)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    last_error = None
    for _ in range(2):
        response = _call(
            model, messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(content)
        except ValidationError as e:
            last_error = e
            logger.warning("llm sortie invalide (%s) : %s", schema.__name__, e.errors()[:2])
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Sortie invalide : {e.errors()[:3]}. Corrige le JSON."},
            ]
    raise LLMError(f"Sortie invalide après un nouvel essai ({schema.__name__})") from last_error


def with_tools(messages: list[dict], tools: list[dict], model: str | None = None):
    """Run one agent turn where the model must call one tool at a time.

    The model sees each tool result before choosing the next action,
    preventing blind parallel queries that consume the budget.
    Returns the assistant message containing the tool call.
    """
    response = _call(
        model or config.MODEL_AGENT,
        messages=messages, tools=tools, tool_choice="required", parallel_tool_calls=False,
        temperature=0,
    )
    message = response.choices[0].message
    if not message.tool_calls:
        raise LLMError("Le modèle n'a appelé aucun outil")
    return message

