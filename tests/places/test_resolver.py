"""Place resolution."""

import pytest

from soretrak.config import ROOT

from soretrak.contracts import Candidate
from soretrak.places.normalize import normalize
from soretrak.places.resolver import get_index, resolve_place


@pytest.mark.parametrize(
    "raw, expected",
    [
        # case, accents, punctuation
        ("Bab Jédid", "bab jedid"),
        ("BIR JEDID-Bouhajla", "bir jedid bouhajla"),
        ("KEF/MAKTHAR", "kef makthar"),
        ("", ""),
        # Arabic → Latin, article ال
        ("باب جديد", "bab jdid"),
        ("باب الجديد", "bab jdid"),
        # arabizi, without touching real numbers
        ("7ammam lif", "hammam lif"),
        ("9ayrawen", "qayrawen"),
        ("ligne 21", "ligne 21"),
        ("kilometre 17", "kilometre 17"),
        ("Ecole 02 Mars 1934", "ecole 02 mars 1934"),
        # separate, attached, assimilated and elided articles
        ("EL FAHS", "fahs"),
        ("ELFAHS", "fahs"),
        ("ESSAIED", "saied"),
        ("el saied", "saied"),
        ("ERRAHBA", "rahba"),
        ("ECHATT", "chatt"),
        ("EJBIL", "jbil"),
        ("Ecole Beit Al Hikma", "ecole beit hikma"),
        ("STADE L'AFH", "stade afh"),
        ("D. EL M'SALLA", "d msalla"),
        # "el" + vowel and ordinary e- words are left untouched
        ("TRANSPORT ELEVES", "transport eleves"),
        ("ENFIDHA", "enfidha"),
        ("ESTOUH", "estouh"),
        # unambiguous abbreviations only
        ("KAI. S.SAAD", "kairouan sidi saad"),
        ("KAIR. JLOULA", "kairouan jloula"),
        ("S..SAAD", "sidi saad"),
        ("M.ALI", "mohamed ali"),
        ("Med Ali", "mohamed ali"),
        # ambiguous abbreviations left as they are
        ("B. ESSAHEL", "b sahel"),
        ("O.BOUZID", "o bouzid"),
        ("K.BOUDHADI", "k boudhadi"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


ORIGINAL_DB = ROOT / "data" / "soretrak_gtfs.db"


@pytest.fixture
def index():
    if not ORIGINAL_DB.exists():
        pytest.skip("data/soretrak_gtfs.db absente — lancer scripts/build_db.py d'abord")
    return get_index(ORIGINAL_DB)


def test_index_covers_all_stops_and_routes(index):
    assert sum(e.type == "arret" for e in index) == 192
    assert sum(e.type == "ligne" for e in index) == 95


def test_index_keeps_raw_names(index):
    bab = next(e for e in index if e.type == "arret" and e.id == "432")
    assert bab.name == "BAB JEDID"
    assert bab.norm == "bab jedid"


def resolve(texte, **kwargs):
    if not ORIGINAL_DB.exists():
        pytest.skip("data/soretrak_gtfs.db absente — lancer scripts/build_db.py d'abord")
    return resolve_place(texte, db_path=ORIGINAL_DB, **kwargs)


@pytest.mark.parametrize("texte", ["21", "ligne 21", "bus21", "Ligne 21", "la 21"])
def test_line_number(texte):
    result = resolve(texte)
    assert result.statut == "trouve"
    assert (result.candidats[0].type, result.candidats[0].id) == ("ligne", "21")


def test_unknown_line_number_is_not_found():
    assert resolve("ligne 99").statut == "introuvable"


def test_number_in_stop_name_is_not_a_line():
    assert resolve("21", type="arret").statut == "introuvable"
    result = resolve("numero 10")
    assert result.candidats[0].nom == "NUMERO 10"


@pytest.mark.parametrize(
    "texte, stop_id",
    [
        ("sousse", "257"),
        ("bab jdid", "432"),
        ("باب جديد", "432"),
        ("kilometre 17", "696"),   # identical name despite kilometre 71 / 74
        ("saied", "370"),          # line names do not interfere
        ("hammem lif", "245"),
    ],
)
def test_found(texte, stop_id):
    result = resolve(texte)
    assert result.statut == "trouve", result
    assert result.candidats[0].id == stop_id


def test_ambiguous_keeps_raw_names():
    result = resolve("bir jedid")
    assert result.statut == "ambigu"
    names = [c.nom for c in result.candidats]
    assert "Bir Jdid" in names and "BIR JEDID-Bouhajla" in names


@pytest.mark.parametrize("texte", ["paris", "tour eiffel", "", "   "])
def test_not_found(texte):
    result = resolve(texte)
    assert result.statut == "introuvable"
    assert result.candidats == []


def test_confirmed_entity_wins_over_ambiguity():
    confirmed = Candidate(type="arret", id="227", nom="BIR JEDID-Bouhajla", score=90, raison="")
    result = resolve("bir jedid", entites_confirmees=[confirmed])
    assert result.statut == "trouve"
    assert result.candidats[0].id == "227"


def test_confirmed_entity_does_not_hijack_a_clear_match():
    # a confirmed Bir Jdid does not prevent "sousse" from finding SOUSSE
    confirmed = Candidate(type="arret", id="579", nom="Bir Jdid", score=90, raison="")
    result = resolve("sousse", entites_confirmees=[confirmed])
    assert result.candidats[0].id == "257"


def test_line_names_only_searched_with_type_ligne():
    assert all(c.type == "arret" for c in resolve("sousse").candidats)
    assert all(c.type == "ligne" for c in resolve("sousse", type="ligne").candidats)


def test_at_most_five_candidates():
    assert len(resolve("sousse", type="ligne").candidats) <= 5


def test_falls_back_to_line_names_when_no_stop_matches():
    result = resolve("LAMINIA MANSOURA")
    assert result.statut == "trouve"
    assert (result.candidats[0].type, result.candidats[0].id) == ("ligne", "1")


def test_word_level_match_tolerates_one_letter():
    names = [c.nom for c in resolve("mansoura").candidats]
    assert "Immeubles El Mansourah" in names and "FINANCES MANSOURAH - ALLER" in names


def test_stop_match_still_wins_over_line_names():
    assert all(c.type == "arret" for c in resolve("saied").candidats)



def test_place_facts():
    from soretrak.places.facts import place_facts
    line = resolve("21").candidats[0]
    facts = place_facts(line, ORIGINAL_DB)
    assert facts["a_horaires"] is True and facts["nb_arrets"] > 0
    stop = resolve("sousse").candidats[0]
    assert place_facts(stop, ORIGINAL_DB)["lignes"] == ["311", "768", "790"]
    line5 = resolve("ligne 5").candidats[0]
    assert place_facts(line5, ORIGINAL_DB)["a_horaires"] is False


def test_resolve_mentions_dedupes_and_keeps_statuses():
    from soretrak.places.facts import resolve_mentions
    if not ORIGINAL_DB.exists():
        pytest.skip("base absente")
    result = resolve_mentions(["bab jdid", "bab jdid ", "paris", ""], db_path=ORIGINAL_DB)
    assert [(r["texte"], r["statut"]) for r in result] == [("bab jdid", "trouve"), ("paris", "introuvable")]
