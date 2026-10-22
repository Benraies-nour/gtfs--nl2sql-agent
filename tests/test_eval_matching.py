"""Tests of the evaluation checker (evals/matching.py).

No LLM call: matching.py only imports the standard library.
The answers used here are the ones actually obtained during the first
evaluation run; they pin down the two false negatives that were fixed.
"""

from pathlib import Path

import pytest
import yaml

from evals.matching import correspond, evalue, mots_manquants, normaliser, valeurs_reference

ANY = "any"
CASES_E2E = yaml.safe_load(
    (Path(__file__).parent.parent / "evals" / "cases_e2e.yaml").read_text(encoding="utf-8"))
WITH_REFERENCE = [c for c in CASES_E2E if c.get("reference")]
CASES_ROUTER = yaml.safe_load(
    (Path(__file__).parent.parent / "evals" / "cases_router.yaml").read_text(encoding="utf-8"))

# --------------------------------------------------------------------------
# normaliser
# --------------------------------------------------------------------------


def test_normaliser_minuscules_et_sans_accents():
    assert normaliser("Bâb Jédid") == "bab jedid"
    assert normaliser("BAB JEDID") == "bab jedid"


def test_normaliser_reduit_les_espaces():
    assert normaliser("  Kairouan   à  Kef  ") == "kairouan a kef"


# --------------------------------------------------------------------------
# correspond: accents and case
# --------------------------------------------------------------------------


def test_correspond_insensible_aux_accents():
    assert correspond("Il y a 1 arrêt à Bâb Jédid", "BAB JEDID") is True


def test_correspond_insensible_a_la_casse():
    assert correspond("l'arret bab jedid", "BAB JEDID") is True


# --------------------------------------------------------------------------
# correspond: numeric values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reponse", ["45 passages", "125 arrêts", "le nombre 1 253", "arrêts n°12 et 13"]
)
def test_le_chiffre_1_ne_matche_pas_un_prefixe_de_nombre(reponse):
    assert correspond(reponse, "1") is False


def test_le_chiffre_1_matche_un_nombre_isole():
    assert correspond("Il y a 1 arrêt", "1") is True


def test_le_chiffre_1_matche_entoure_de_ponctuation():
    assert correspond("Réponse : 1.", "1") is True


def test_le_chiffre_1_matche_apres_un_symbole_non_chiffre():
    """n°1 is indeed the number 1: ° is an ordinal sign, not a digit."""
    assert correspond("arrêt n°1 sur 3", "1") is True


def test_une_valeur_numerique_ne_passe_jamais_par_sous_chaine():
    """Regression: the substring check came before the word boundary, so
    "1" matched "125". The order of the strategies matters."""
    assert correspond("il y a 125 arrêts", "1") is False
    assert correspond("il y a 12 arrêts", "2") is False


# --------------------------------------------------------------------------
# correspond: composite values
# --------------------------------------------------------------------------


def test_valeur_composite_tous_les_mots_presents():
    assert correspond("La ligne 2003 relie Kairouan à Kef.", "KAIROUAN KEF") is True


def test_valeur_composite_un_seul_mot_manquant_ne_passe_pas():
    assert correspond("La ligne 2003 part de Kairouan.", "KAIROUAN KEF") is False
    assert mots_manquants("La ligne 2003 part de Kairouan.", "KAIROUAN KEF") == ["kef"]


def test_valeur_vide_ne_correspond_jamais():
    assert correspond("n'importe quoi", "") is False


# --------------------------------------------------------------------------
# evalue: any / all modes
# --------------------------------------------------------------------------


def test_mode_any_accepte_une_seule_valeur():
    reussi, trouvees, _ = evalue("La ligne est NASRALLAH", ["430", "NASRALLAH"], ANY)
    assert reussi is True
    assert trouvees == ["NASRALLAH"]


def test_mode_any_echoue_si_aucune_valeur():
    reussi, trouvees, _ = evalue("Le bus 999", ["430", "NASRALLAH"], ANY)
    assert reussi is False
    assert trouvees == []


def test_mode_all_exige_toutes_les_valeurs():
    assert evalue("430 et NASRALLAH", ["430", "NASRALLAH"], "all")[0] is True
    assert evalue("seulement 430", ["430", "NASRALLAH"], "all")[0] is False


def test_mode_all_sur_liste_vide_ne_passe_pas():
    """A case whose reference is empty must never pass."""
    assert evalue("une reponse quelconque", [], "all")[0] is False
    assert evalue("une reponse quelconque", [], ANY)[0] is False


# --------------------------------------------------------------------------
# The six checks of the former self-test, on real answers
# --------------------------------------------------------------------------


def test_2003_reponse_correcte_reformulee_passe():
    assert correspond("La ligne 2003 relie Kairouan à Kef.", "KAIROUAN KEF") is True


def test_elargoub_reponse_par_route_long_name_passe():
    assert correspond("Le bus NASRALLAH est le seul à passer par l'arrêt B. ELARGOUB.", "NASRALLAH") is True


def test_elargoub_reponse_par_route_short_name_passe():
    assert correspond("Seule la ligne 430 dessert cet arrêt.", "430") is True


def test_bab_jedid_confusion_arrets_passages_echoue():
    reponse = "Il y a 45 passages enregistrés dont l'enseigne de destination contient 'BAB JEDID'."
    assert evalue(reponse, ["1"], "all")[0] is False


def test_bab_jedid_confusion_arrets_lignes_echoue():
    reponse = "7 lignes passent par Bab Jedid."
    assert evalue(reponse, ["1"], "all")[0] is False


def test_bab_jedid_vraie_reponse_passe():
    assert evalue("Il y a 1 arrêt qui s'appelle BAB JEDID.", ["1"], "all")[0] is True


# --------------------------------------------------------------------------
# The evaluation case files are well formed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cas", WITH_REFERENCE, ids=lambda c: c["id"])
def test_chaque_reference_renvoie_des_valeurs(cas):
    """An empty reference would make the PASS vacuous: it would prove nothing."""
    attendus = valeurs_reference(cas["reference"], cas.get("colonnes", [0]))
    assert attendus, f"cas {cas['id']} : la requete de reference ne renvoie aucune valeur"


def test_cas_e2e_bien_formes():
    chemins = {"repondu", "aucune_donnee", "ambigu", "impossible", "conversation",
               "trop_vague", "hors_perimetre", "hors_sujet"}
    ids = [c["id"] for c in CASES_E2E]
    assert len(ids) == len(set(ids)), "ids en double"
    for cas in CASES_E2E:
        attendus = cas["attendu"] if isinstance(cas["attendu"], list) else [cas["attendu"]]
        assert set(attendus) <= chemins, cas["id"]
        if attendus == ["repondu"]:
            assert cas.get("reference"), f"{cas['id']} : repondu sans reference"


def test_cas_router_bien_formes():
    intents = {"transport", "conversation", "trop_vague", "hors_perimetre", "hors_sujet"}
    assert all(c["intent"] in intents for c in CASES_ROUTER)
    assert {c["intent"] for c in CASES_ROUTER} == intents
