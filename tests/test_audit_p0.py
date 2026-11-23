"""Tests de non-régression — corrections audit 20/11/2026."""
import numpy as np
import pandas as pd
import pytest

from icv import double_peine, compute_icv, normalize, prioritization_quadrant


# ═══════════════════════════════════════════════════════════════
# TESTS double_peine()
# ═══════════════════════════════════════════════════════════════
def test_double_peine_identifiee():
    df = pd.DataFrame({
        "commune": ["A"],
        "score_sante": [70], "score_eau": [70], "score_alimentation": [70],
        "score_inondation": [60],
    })
    r = double_peine(df)
    assert r["double_peine"].iloc[0] == True
    assert r["double_peine_statut"].iloc[0] == "Double peine identifiée"


def test_double_peine_inondation_nan_pas_zero():
    """AUD-01 : une inondation NaN ne doit JAMAIS être traitée comme 0."""
    df = pd.DataFrame({
        "commune": ["B"],
        "score_sante": [70], "score_eau": [70], "score_alimentation": [70],
        "score_inondation": [np.nan],
    })
    r = double_peine(df)
    assert r["double_peine"].iloc[0] == False
    assert r["double_peine_statut"].iloc[0] == "Données insuffisantes"


def test_double_peine_services_insuffisants():
    """Un seul pilier service documenté → données insuffisantes."""
    df = pd.DataFrame({
        "commune": ["C"],
        "score_sante": [70], "score_eau": [np.nan], "score_alimentation": [np.nan],
        "score_inondation": [60],
    })
    r = double_peine(df)
    assert r["double_peine_statut"].iloc[0] == "Données insuffisantes"


def test_double_peine_non_identifiee():
    df = pd.DataFrame({
        "commune": ["D"],
        "score_sante": [30], "score_eau": [30], "score_alimentation": [30],
        "score_inondation": [30],
    })
    r = double_peine(df)
    assert r["double_peine_statut"].iloc[0] == "Non identifiée"


# ═══════════════════════════════════════════════════════════════
# TESTS compute_icv()
# ═══════════════════════════════════════════════════════════════
def test_icv_couverture_complete():
    df = pd.DataFrame({
        "commune": ["X"],
        "score_sante": [70], "score_eau": [60],
        "score_alimentation": [80], "score_inondation": [50],
        "score_erosion": [40],
    })
    r = compute_icv(df, {p: 0.2 for p in ["sante", "eau", "alimentation",
                                           "inondation", "erosion"]})
    assert r["statut_icv"].iloc[0] == "DISPONIBLE"
    assert r["couverture_icv_pct"].iloc[0] == 100.0


def test_icv_donnees_insuffisantes():
    """Couverture < 75 % → pas de classement."""
    df = pd.DataFrame({
        "commune": ["Y"],
        "score_sante": [70], "score_eau": [np.nan],
        "score_alimentation": [np.nan], "score_inondation": [np.nan],
        "score_erosion": [np.nan],
    })
    r = compute_icv(df, {p: 0.2 for p in ["sante", "eau", "alimentation",
                                           "inondation", "erosion"]})
    assert r["statut_icv"].iloc[0] == "INDISPONIBLE"
    assert pd.isna(r["ICV"].iloc[0])


def test_normalize_direction_bornage_nan_infini():
    s = pd.Series([0.0, 50.0, 100.0, np.nan, np.inf])
    positive = normalize(s, "positive")
    negative = normalize(s, "negative")
    assert positive.dropna().between(0, 100).all()
    assert negative.dropna().between(0, 100).all()
    assert pd.isna(positive.iloc[-1])
    assert pd.isna(negative.iloc[-1])


def test_normalize_constante_non_borneee():
    r = normalize(pd.Series([5.0, 5.0, 5.0]), "positive")
    assert r.isna().all()


def test_prioritization_quadrant_and_missing_values():
    zones = prioritization_quadrant(
        pd.Series([60, 65, 50, 50, np.nan]),
        pd.Series([55, 50, 60, 50, 60]),
    )
    assert zones.tolist() == [
        "Urgence",
        "À équiper",
        "À protéger",
        "Suivi",
        "Données insuffisantes",
    ]
