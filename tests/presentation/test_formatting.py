"""Formatting of results: times, directions, technical columns, readable names."""
from soretrak.presentation.formatting import (clean_rows, clean_text, human, keep_line_breaks,
                                               readable_rows)


def test_synthesizer_cleans_seconds_and_technical_columns():
    rows = [{"trip_id": "8", "direction_id": 0, "route_id": "21", "departure_time": "06:30:00"}]
    assert clean_rows(rows) == [{"sens": "aller", "route_id": "21", "departure_time": "06:30"}]
    assert clean_text("Aucun départ après 18:51:38. Sens direction_id = 0 (CNR).") == \
        "Aucun départ après 18:51. (CNR)."
    assert clean_text("Ligne 21 (direction_id = 0)") == "Ligne 21"


def test_synthesizer_cleans_plain_sens_mention():
    assert clean_text("Horaires de la ligne 21 (sens 0).") == "Horaires de la ligne 21."
    assert clean_text("Sens 1 vers Sousse") == " vers Sousse"
    assert clean_text("les sens uniques") == "les sens uniques"


def test_rows_of_several_lines_are_grouped_by_line_keeping_time_order():
    rows = [{"route_id": "21", "departure_time": "06:30:00"},
            {"route_id": "70", "departure_time": "06:45:00"},
            {"route_id": "21", "departure_time": "07:20:00"},
            {"route_id": "70", "departure_time": "07:22:00"},
            {"route_id": "22", "departure_time": "08:30:00"}]
    grouped = clean_rows(rows)
    assert [(r["route_id"], r["departure_time"]) for r in grouped] == [
        ("21", "06:30"), ("21", "07:20"), ("70", "06:45"), ("70", "07:22"), ("22", "08:30")]


def test_single_line_rows_keep_their_order():
    rows = [{"route_id": "21", "departure_time": "07:20:00"}, {"route_id": "21", "departure_time": "06:30:00"}]
    assert [r["departure_time"] for r in clean_rows(rows)] == ["07:20", "06:30"]


def test_human_column_names():
    assert human("nb_arrets") == "Nombre d'arrêts"
    assert human("nombre_departs") == "Nombre de départs"
    assert human("n") == "Nombre"


def test_readable_rows_uses_plain_column_names():
    rows = [{"route_id": "21", "stop_name": "BAB JEDID", "departure_time": "06:30:00",
             "direction_id": 0, "trip_id": "8", "nb_departs": 3}]
    assert readable_rows(rows) == [{"Ligne": "21", "Arrêt": "BAB JEDID", "Heure": "06:30",
                                    "Sens": "aller", "Nombre de départs": 3}]


def test_keep_line_breaks_separates_list_lines():
    assert keep_line_breaks("Voici :\nligne 3\nligne 22") == "Voici :  \nligne 3  \nligne 22"


def test_keep_line_breaks_leaves_tables_intact_and_surrounds_them_with_blank_lines():
    text = "Horaires :\n| Ligne | Départs |\n|---|---|\n| 21 | 06:30 |\nThéoriques."
    assert keep_line_breaks(text) == \
        "Horaires :\n\n| Ligne | Départs |\n|---|---|\n| 21 | 06:30 |\n\nThéoriques."
