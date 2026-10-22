"""On-demand map
"""
from pathlib import Path

from soretrak.config import DB_PATH
from soretrak.contracts import MapData, ResolvedPlace
from soretrak.sql.executor import run_query


def _q(sql: str, db: str | Path, params: tuple = ()) -> list[dict]:
    result = run_query(sql, db, max_rows=5000, params=params)
    return result.rows if result.ok else []


def _point(row: dict, role: str) -> dict:
    return {"nom": row["stop_name"], "lat": row["stop_lat"], "lon": row["stop_lon"], "role": role}


def _route_stops(route_id: str, direction: int, db) -> list[dict]:
    return _q(
        "SELECT rs.stop_sequence, s.stop_id, s.stop_name, s.stop_lat, s.stop_lon "
        "FROM v_route_stops rs JOIN stops s ON s.stop_id = rs.stop_id "
        "WHERE rs.route_id = ? AND rs.direction_id = ? "
        "ORDER BY rs.stop_sequence", db, (route_id, direction))


def _shape(route_id: str, direction: int, db) -> list[list[float]]:
    """GPS points of the line's route ([lon, lat]), empty if the line has none."""
    rows = _q(
        "SELECT sh.shape_pt_lat AS lat, sh.shape_pt_lon AS lon FROM shapes sh "
        "WHERE sh.shape_id = (SELECT shape_id FROM trips "
        "WHERE route_id = ? AND direction_id = ? LIMIT 1) "
        "ORDER BY sh.shape_pt_sequence", db, (route_id, direction))
    return [[r["lon"], r["lat"]] for r in rows]


def _nearest(path: list[list[float]], lat: float, lon: float) -> int:
    return min(range(len(path)),
               key=lambda i: (path[i][0] - lon) ** 2 + (path[i][1] - lat) ** 2)


def _path(route_id: str, direction: int, stops: list[dict], db) -> tuple[list, str]:
    """The GPS route between the first and last stop, otherwise the stops joined."""
    shape = _shape(route_id, direction, db)
    if shape:
        a = _nearest(shape, stops[0]["stop_lat"], stops[0]["stop_lon"])
        b = _nearest(shape, stops[-1]["stop_lat"], stops[-1]["stop_lon"])
        segment = shape[min(a, b):max(a, b) + 1]
        if len(segment) >= 2:
            return segment, "gps"
    return [[s["stop_lon"], s["stop_lat"]] for s in stops], "approx"


def _direct(a: str, b: str, db) -> dict | None:
    """A line that serves A then B (same direction); preferably with a GPS route,
    then with the fewest stops between them."""
    rows = _q(
        "SELECT a.route_id, a.direction_id, a.stop_sequence AS sa, b.stop_sequence AS sb, "
        "EXISTS (SELECT 1 FROM trips t JOIN shapes sh ON sh.shape_id = t.shape_id "
        "        WHERE t.route_id = a.route_id AND t.direction_id = a.direction_id) AS gps "
        "FROM v_route_stops a JOIN v_route_stops b "
        "ON a.route_id = b.route_id AND a.direction_id = b.direction_id "
        "WHERE a.stop_id = ? AND b.stop_id = ? AND a.stop_sequence < b.stop_sequence "
        "ORDER BY gps DESC, b.stop_sequence - a.stop_sequence LIMIT 1", db, (a, b))
    return rows[0] if rows else None


def _stop(stop_id: str, db) -> dict | None:
    rows = _q("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops WHERE stop_id = ?", db, (stop_id,))
    return rows[0] if rows else None


def build_map(lieux_resolus: list[ResolvedPlace], db_path: str | Path | None = None) -> MapData | None:
    """Map data for the places found, or None if there is nothing to show."""
    db = db_path or DB_PATH
    found = [lieu["candidats"][0] for lieu in lieux_resolus
             if lieu["statut"] == "trouve" and lieu["candidats"]]
    stops = [c for c in found if c["type"] == "arret"]
    lines = [c for c in found if c["type"] == "ligne"]

    if len(stops) >= 2:
        a, b = stops[0], stops[1]
        for start, end in ((a, b), (b, a)):
            link = _direct(start["id"], end["id"], db)
            if link:
                seq = [s for s in _route_stops(link["route_id"], link["direction_id"], db)
                       if link["sa"] <= s["stop_sequence"] <= link["sb"]]
                path, trace = _path(link["route_id"], link["direction_id"], seq, db)
                points = [_point(s, "intermediaire") for s in seq[1:-1]]
                points += [_point(seq[0], "depart"), _point(seq[-1], "arrivee")]
                return {"titre": f"Ligne {link['route_id']} : {seq[0]['stop_name']} → {seq[-1]['stop_name']}",
                        "points": points, "chemin": path, "trace": trace, "note": None}
        points = [_point(s, role) for s, role in ((_stop(a["id"], db), "depart"),
                                                    (_stop(b["id"], db), "arrivee")) if s]
        return {"titre": f"{a['nom']} et {b['nom']}", "points": points, "chemin": [],
                "trace": None, "note": "Pas de ligne directe entre ces deux arrêts dans les données."}

    if lines:
        line = lines[0]
        for direction in (0, 1):
            seq = _route_stops(line["id"], direction, db)
            if seq:
                path, trace = _path(line["id"], direction, seq, db)
                points = [_point(s, "intermediaire") for s in seq[1:-1]]
                points += [_point(seq[0], "depart"), _point(seq[-1], "arrivee")]
                return {"titre": f"Ligne {line['id']} : {line['nom']}", "points": points,
                        "chemin": path, "trace": trace, "note": None}
        return {"titre": f"Ligne {line['id']} : {line['nom']}", "points": [], "chemin": [],
                "trace": None, "note": "Les arrêts de cette ligne ne figurent pas dans les données."}

    if stops:
        stop = _stop(stops[0]["id"], db)
        if stop:
            return {"titre": stop["stop_name"], "points": [_point(stop, "arret")], "chemin": [],
                    "trace": None, "note": None}
    return None


_ROLE_COLORS = {"depart": "#22a050", "arrivee": "#d63031", "arret": "#d63031",
                "intermediaire": "#7a7a7a"}


def to_leaflet_html(carte: MapData, height: int = 420) -> str:
    """Leaflet HTML page (OpenStreetMap tiles): GPS route points joined in order,
    departure, arrival and intermediate stops. No API key, no Python dependency."""
    import json

    data = {
        # shapes: [lon, lat] → Leaflet expects [lat, lon]
        "chemin": [[lat, lon] for lon, lat in carte["chemin"]],
        "gps": carte["trace"] == "gps",
        "points": [{**p, "couleur": _ROLE_COLORS.get(p["role"], "#7a7a7a")} for p in carte["points"]],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", r"<\/")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body{{margin:0}}#map{{height:{height}px;border-radius:12px}}</style>
</head><body><div id="map"></div><script>
const d = {payload};
const map = L.map('map', {{scrollWheelZoom: false}});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 19, attribution: '&copy; OpenStreetMap'}}).addTo(map);
const bounds = [];
if (d.chemin.length > 1) {{
  L.polyline(d.chemin, d.gps ? {{color: '#1e6edc', weight: 4}}
                             : {{color: '#f0961e', weight: 3, dashArray: '6 8'}}).addTo(map);
  if (d.gps) d.chemin.forEach(p => L.circleMarker(p, {{radius: 2, weight: 0,
      fillColor: '#1e6edc', fillOpacity: 0.9}}).addTo(map));
  bounds.push(...d.chemin);
}}
d.points.forEach(p => {{
  const big = p.role !== 'intermediaire';
  L.circleMarker([p.lat, p.lon], {{radius: big ? 9 : 5, color: '#fff', weight: 2,
      fillColor: p.couleur, fillOpacity: 1}}).bindTooltip(p.nom).addTo(map);
  bounds.push([p.lat, p.lon]);
}});
if (bounds.length > 1) map.fitBounds(bounds, {{padding: [30, 30]}});
else if (bounds.length === 1) map.setView(bounds[0], 15);
</script></body></html>"""
