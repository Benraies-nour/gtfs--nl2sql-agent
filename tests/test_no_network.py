"""Test of the "no network call during tests" guarantee.

The `bloquer_reseau` fixture in conftest.py must make any accident
impossible: a real call must fail immediately, with a message naming
the offending test.
"""

import socket
import urllib.request

import pytest

from conftest import ReseauBloque


def test_un_vrai_appel_httpe_est_bloque():
    """A real call must raise ReseauBloque, not go through."""
    with pytest.raises(ReseauBloque) as info:
        urllib.request.urlopen("https://api.openai.com/v1/models", timeout=5)
    message = str(info.value)
    assert "test_un_vrai_appel_httpe_est_bloque" in message, (
        "le message doit nommer le test fautif pour etre actionnable"
    )
    assert "permet_reseau" in message, "le message doit indiquer la porte de sortie"


def test_socket_connect_est_bloque():
    with pytest.raises(ReseauBloque):
        socket.create_connection(("api.groq.com", 443), timeout=5)


def test_socket_connect_ex_est_bloque():
    """connect_ex is cut too, otherwise an SDK could get around the block."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(ReseauBloque):
            s.connect_ex(("api.groq.com", 443))
    finally:
        s.close()


def test_erreur_nomme_cequi_a_ete_cible():
    """The error must show the target to be diagnosable."""
    with pytest.raises(ReseauBloque) as info:
        socket.create_connection(("exemple.invalid", 443))
    assert "exemple.invalid" in str(info.value)


def test_construction_de_socket_reste_possible():
    """Only opening is blocked: the rest of the test must remain usable."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()  # no exception: only connect/connect_ex are cut


def test_llm_vrai_appel_serait_bloque(monkeypatch):
    """Targeted proof: a real call to soretrak.llm, without mocks, must be blocked.

    The gateway turns the connection failure into an LLMError; going up the
    chain, the original cause is indeed the conftest block.
    """
    from soretrak import config, llm
    from soretrak.contracts import RouterDecision

    monkeypatch.setattr(config, "LLM_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setitem(config.PROVIDERS[config.LLM_PROVIDER], "api_key", "cle-de-test")
    llm.set_client(None)
    try:
        with pytest.raises(llm.LLMError) as info:
            llm.structured("systeme", "question", RouterDecision)
    finally:
        llm.set_client(None)

    cause, chaine = info.value, []
    while cause is not None:
        chaine.append(cause)
        cause = cause.__cause__ or cause.__context__
    assert any(isinstance(c, ReseauBloque) for c in chaine), chaine
