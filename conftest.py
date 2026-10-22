"""Repository-wide pytest configuration.

Two guarantees are provided here:

1. No test may call a real API. A forgotten mock or a configuration
   regression would cost real quota. The `bloquer_reseau` fixture cuts
   connections and raises an explicit error naming the offending test.
2. `evals/run.py` calls the real LLM: it must stay out of collection.
"""

import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
collect_ignore = [str(ROOT / "evals" / "run.py")]


class ReseauBloque(RuntimeError):
    """A test tried to reach the network."""


@pytest.fixture(autouse=True)
def bloquer_reseau(request, monkeypatch):
    """Forbid any network connection during a test.

    Only the connection primitives are cut: creating a socket still works,
    only opening it to the outside fails. Without this, an SDK built at
    import time would try to go out.
    """
    if request.node.get_closest_marker("permet_reseau"):
        yield
        return

    def refuse(cible, *args, **kwargs):
        raise ReseauBloque(
            f"Le test '{request.node.name}' a tente une connexion reseau vers "
            f"{cible!r}. Les tests ne doivent pas appeler d'API : "
            f"mockez le client LLM, ou marquez le test avec @pytest.mark.permet_reseau "
            f"si l'appel est volontaire."
        )

    monkeypatch.setattr(socket.socket, "connect", refuse, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse, raising=True)
    monkeypatch.setattr(socket, "create_connection", refuse, raising=True)
    yield
