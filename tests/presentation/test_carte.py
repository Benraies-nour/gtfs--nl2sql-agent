"""On-demand map: built by the code from the resolved places (real database)."""

import pytest

from soretrak.config import ROOT

from soretrak.presentation.carte import build_map
from soretrak.places.facts import resolve_mentions

DB = ROOT / "data" / "soretrak_gtfs.db"


@pytest.fixture(autouse=True)
def _db():
    if not DB.exists():
        pytest.skip("data/soretrak_gtfs.db absente")


def carte(*lieux):
    return build_map(resolve_mentions(list(lieux), db_path=DB), DB)


def roles(m):
    return [p["role"] for p in m["points"]]


def test_two_stops_prefers_direct_line_with_gps_shape():
    m = carte("Immeubles El Mansourah", "bab jdid")
    assert m["titre"].startswith("Ligne 22")
    assert m["trace"] == "gps" and len(m["chemin"]) > 10
    assert roles(m).count("depart") == 1 and roles(m).count("arrivee") == 1
    depart = next(p for p in m["points"] if p["role"] == "depart")
    assert depart["nom"] == "Immeubles El Mansourah"


def test_two_stops_without_gps_uses_stops_in_order():
    m = carte("Immeubles El Mansourah", "X - SOUSSE")
    assert m["titre"].startswith("Ligne 21")
    assert m["trace"] == "approx"
    assert len(m["chemin"]) == len(m["points"])


def test_order_of_mention_does_not_matter():
    m = carte("X - SOUSSE", "Immeubles El Mansourah")
    assert m["titre"] == "Ligne 21 : Immeubles El Mansourah → X - SOUSSE"


def test_no_direct_line_shows_both_stops_with_note():
    m = carte("bab jdid", "sousse")
    assert m["chemin"] == [] and len(m["points"]) == 2
    assert "Pas de ligne directe" in m["note"]


def test_line_with_gps_shape():
    m = carte("311")
    assert m["trace"] == "gps" and len(m["points"]) == 10


def test_line_without_stops_has_note():
    m = carte("5")
    assert m["points"] == [] and "ne figurent pas" in m["note"]


def test_single_stop():
    m = carte("bab jdid")
    assert [p["nom"] for p in m["points"]] == ["BAB JEDID"] and m["chemin"] == []


def test_ambiguous_or_missing_place_gives_no_map():
    assert carte("bir jedid") is None
    assert carte("paris") is None
    assert build_map([], DB) is None



def test_leaflet_html_draws_shape_points_in_lat_lon_order():
    from soretrak.presentation.carte import to_leaflet_html
    m = carte("Immeubles El Mansourah", "bab jdid")
    html = to_leaflet_html(m)
    lon, lat = m["chemin"][0]
    assert f"[{lat}, {lon}]" in html                 # Leaflet: [lat, lon]
    assert "tile.openstreetmap.org" in html and "L.polyline" in html
    assert "</script></body></html>" in html


def test_leaflet_html_escapes_closing_tags():
    from soretrak.presentation.carte import to_leaflet_html
    m = {"chemin": [], "trace": None, "titre": "x",
         "points": [{"nom": "</script><b>", "lat": 35.6, "lon": 10.1, "role": "arret"}]}
    assert "</script><b>" not in to_leaflet_html(m)
