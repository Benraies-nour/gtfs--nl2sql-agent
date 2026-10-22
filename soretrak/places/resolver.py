"""Place resolution: user text → stop or line.

"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz

from soretrak.config import DB_PATH
from soretrak.contracts import Candidate, PlaceResult, PlaceType
from soretrak.places.normalize import normalize
from soretrak.sql.executor import run_query


@dataclass(frozen=True)
class IndexEntry:
    type: str       # "arret" or "ligne"
    id: str         # stop_id or route_id
    name: str       # raw name from the database, shown as is to the agent
    norm: str


def _build_index(db_path: str | Path) -> list[IndexEntry]:
    """Read the stops and lines from the database and prepare their search forms."""
    stops = run_query("SELECT stop_id, stop_name FROM stops", db_path, max_rows=10_000)
    routes = run_query(
        "SELECT route_id, route_long_name FROM routes", db_path, max_rows=10_000
    )
    for result in (stops, routes):
        if not result.ok:
            raise RuntimeError(f"Index des lieux impossible à construire : {result.error}")

    entries = []
    for type_, result, id_col, name_col in (
        ("arret", stops, "stop_id", "stop_name"),
        ("ligne", routes, "route_id", "route_long_name"),
    ):
        for row in result.rows:
            name = row[name_col] or ""
            entries.append(IndexEntry(type_, str(row[id_col]), name, normalize(name)))
    return entries


@lru_cache(maxsize=4)
def get_index(db_path: str | Path) -> tuple[IndexEntry, ...]:
    """Index built once per database, then kept in memory."""
    return tuple(_build_index(db_path))


# Decision thresholds, tuned with evals/cases_places.yaml.
FOUND_SCORE = 85
FOUND_GAP = 10
AMBIGUOUS_SCORE = 70
MAX_CANDIDATES = 5

# A name contained in a longer one ("sousse" in "X - SOUSSE").
CONTAINED_WEIGHT = 0.9

_GENERIC_PREFIX = re.compile(r"^(?:la |le )?(?:ligne|route|bus|car|arret|station)\s+(?=[a-z])")

_LINE_NUMBER = re.compile(
    r"^(?:(?:la|le)\s+)?(?:(?:ligne|line|bus|car|numero|num|no|n)\s*°?\s*)?(\d{1,4})$"
)


def _words_score(query: str, norm: str) -> float:
    """Each input word against the closest word of the name (mansoura ≈ mansourah)."""
    names = norm.split()
    if not names:
        return 0.0
    words = query.split()
    return sum(max(fuzz.ratio(w, n) for n in names) for w in words) / len(words)


def _score(query: str, entry: IndexEntry) -> tuple[float, str]:
    exact = fuzz.ratio(query, entry.norm)
    contained = CONTAINED_WEIGHT * max(fuzz.token_set_ratio(query, entry.norm),
                                       _words_score(query, entry.norm))
    return max(
        (exact, "nom exact" if exact == 100 else "orthographe proche"),
        (contained, "nom qui contient le texte"),
    )


def _search(query: str, index, wanted: str) -> list:
    """Close candidates (score ≥ ambiguity threshold), best first."""
    scored = sorted(
        ((*_score(query, e), e) for e in index if e.type == wanted),
        key=lambda item: item[0],
        reverse=True,
    )
    return [(s, r, e) for s, r, e in scored[:MAX_CANDIDATES] if s >= AMBIGUOUS_SCORE]


def _candidate(entry: IndexEntry, score: float, reason: str) -> Candidate:
    return Candidate(type=entry.type, id=entry.id, nom=entry.name,
                     score=round(score), raison=reason)


def resolve_place(
    texte: str,
    type: PlaceType | None = None,
    entites_confirmees: list[Candidate] | None = None,
    db_path: str | Path | None = None,
) -> PlaceResult:
    """User text → stop or line: trouve, ambigu or introuvable.

    `type`: "arret", "ligne" or "ville" (a city is searched among the stops).
    Without a type: line number, otherwise stops. Long line names are only
    searched with type="ligne": they mention places and would blur a stop search.
    """
    index = get_index(db_path or DB_PATH)
    query = normalize(texte)
    # "ligne laminia mansoura", "arret bab jdid": the generic word hurts the search
    # ("ligne 21" is still treated as a line number, below).
    query = _GENERIC_PREFIX.sub("", query)
    if not query:
        return PlaceResult(statut="introuvable")

    # 1. Exact line number: "21", "ligne 21", "bus21", "la 22".
    if type in (None, "ligne"):
        match = _LINE_NUMBER.match(query)
        if match:
            line = next((e for e in index if e.type == "ligne" and e.id == match.group(1)), None)
            if line:
                return PlaceResult(statut="trouve",
                                   candidats=[_candidate(line, 100, "numéro de ligne")])

    # 2. Lexical search. Without a type: stops, then line names
    #    if no stop is close ("laminia mansoura" → line 1).
    close = _search(query, index, "ligne" if type == "ligne" else "arret")
    if not close and type is None:
        close = _search(query, index, "ligne")
    if not close:
        return PlaceResult(statut="introuvable")

    # 3. An already confirmed entity wins if it is among the close candidates.
    top = close[0][0]
    confirmed = {(c.type, c.id) for c in entites_confirmees or []}
    for s, _, e in close:
        if (e.type, e.id) in confirmed and s >= top - FOUND_GAP:
            return PlaceResult(statut="trouve",
                               candidats=[_candidate(e, s, "entité déjà confirmée")])

    # 4. Decision: an identical and unique name, otherwise the thresholds.
    identical = [(s, r, e) for s, r, e in close if e.norm == query]
    second = close[1][0] if len(close) > 1 else 0
    if len(identical) == 1 or (top >= FOUND_SCORE and top - second >= FOUND_GAP):
        best_score, best_reason, best = identical[0] if len(identical) == 1 else close[0]
        return PlaceResult(statut="trouve", candidats=[_candidate(best, best_score, best_reason)])
    # Ambiguous: only candidates close to the best one are proposed.
    near = [(s, r, e) for s, r, e in close if s >= top - FOUND_GAP]
    return PlaceResult(statut="ambigu",
                       candidats=[_candidate(e, s, r) for s, r, e in near])
