"""Measurement of resolve_place on evals/cases_places.yaml.

A case passes if the expected place is among the returned candidates
(or if the status is "introuvable" when nothing is expected).
Run with -s to see the details: pytest tests/places/test_places_eval.py -s
"""

import pytest

from soretrak.config import ROOT
import yaml

from evals.matching import place_hit
from soretrak.places.resolver import resolve_place

CASES = yaml.safe_load((ROOT / "evals" / "cases_places.yaml").read_text(encoding="utf-8"))
DB = ROOT / "data" / "soretrak_gtfs.db"

MIN_HIT_RATE = 0.8  # provisional target, set after the first measurement


def test_hit_rate():
    if not DB.exists():
        pytest.skip("data/soretrak_gtfs.db absente — lancer scripts/build_db.py d'abord")
    hits = 0
    for texte, expected in CASES:
        result = resolve_place(texte, db_path=DB)
        ok = place_hit(expected, result)
        hits += ok
        names = ", ".join(c.nom for c in result.candidats) or "—"
        print(f"{'OK ' if ok else 'RATÉ'} {texte!r:24} → {result.statut:11} {names}")
    rate = hits / len(CASES)
    print(f"\n{hits}/{len(CASES)} = {rate:.0%}")
    assert rate >= MIN_HIT_RATE
