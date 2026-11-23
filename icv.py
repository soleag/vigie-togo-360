"""Calcul de l'Indice Communal de Vulnérabilité — version audit 20/11/2026.

Corrections appliquées :
- AUD-01 : données manquantes ≠ zéro dans double_peine()
- AUD-05 : couverture ICV explicite (min_coverage)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import INDICATORS, PILLARS


def normalize(series: pd.Series, direction: str) -> pd.Series:
    """Min-max 0-100. Note : dépend du périmètre (limite documentée)."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    lo, hi = s.min(), s.max()
    if not np.isfinite(hi - lo) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    norm = (s - lo) / (hi - lo) * 100.0
    return (100.0 - norm) if direction == "positive" else norm


def prioritization_quadrant(
    score_services: pd.Series,
    score_inondation: pd.Series,
    services_threshold: float = 60,
    flood_threshold: float = 55,
) -> pd.Series:
    """Classe les territoires dans les quatre cadrans de la matrice."""
    services = pd.to_numeric(score_services, errors="coerce")
    inondation = pd.to_numeric(score_inondation, errors="coerce")
    missing = services.isna() | inondation.isna()
    return pd.Series(
        np.select(
            [
                missing,
                (services >= services_threshold) & (inondation >= flood_threshold),
                (services >= services_threshold) & (inondation < flood_threshold),
                (services < services_threshold) & (inondation >= flood_threshold),
            ],
            [
                "Données insuffisantes",
                "Urgence",
                "À équiper",
                "À protéger",
            ],
            default="Suivi",
        ),
        index=services.index,
        name="zone_priorisation",
    )


def compute_pillars(gdf):
    """Ajoute score_<pilier> (0-100, 100 = plus vulnérable)."""
    gdf = gdf.copy()
    for pillar in PILLARS:
        inds = [k for k, m in INDICATORS.items()
                if m["pillar"] == pillar and k in gdf.columns]
        if not inds:
            gdf[f"score_{pillar}"] = np.nan
            continue
        sub = pd.DataFrame(
            {ind: normalize(gdf[ind], INDICATORS[ind]["direction"]) for ind in inds},
            index=gdf.index,
        )
        gdf[f"score_{pillar}"] = sub.mean(axis=1, skipna=True)
    return gdf


def compute_icv(gdf, weights: dict, min_coverage: float = 0.75):
    """
    ICV avec statuts explicites à 3 états :
    - DISPONIBLE     : données présentes et exploitables
    - NON_APPLICABLE : le pilier ne concerne pas ce territoire (ex: érosion pour intérieures)
    - INDISPONIBLE   : le pilier devrait s'appliquer mais données manquantes
    
    Le statut de chaque territoire est calculé selon les piliers APPLICABLES
    (l'érosion n'est applicable qu'aux communes littorales).
    """
    gdf = gdf.copy()
    
    if gdf.empty:
        gdf["ICV"] = np.nan
        gdf["classe"] = None
        gdf["piliers_disponibles_nb"] = 0
        gdf["piliers_applicables_nb"] = 0
        gdf["couverture_icv_pct"] = np.nan
        gdf["statut_icv"] = "INDISPONIBLE"
        return gdf
    
    # Charger le référentiel littoral
    from pathlib import Path
    import pandas as pd
    ref_path = Path(__file__).parent / "data" / "essentiel" / "communes_littorales.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        ref["_est_littorale"] = (
            ref["est_littorale"].astype("string").str.strip().str.lower()
            .isin(("true", "1", "oui", "yes", "vrai"))
        )
        littorale_map = dict(zip(
            ref["commune_id"].astype(str).str.strip(),
            ref["_est_littorale"].astype(bool),
        ))
    else:
        littorale_map = {}
    
    expected = [p for p in PILLARS if weights.get(p, 0) > 0]
    
    # Calculer piliers applicables par territoire
    _ids = gdf.get("id", pd.Series([None] * len(gdf))).astype(str)
    _mapped = _ids.map(littorale_map)
    if _mapped.isna().all() and _ids.str.len().max() > 7:
        _mapped = _ids.str[:6].map(littorale_map)
    gdf["_est_littorale"] = _mapped.astype("boolean").fillna(False).astype(bool)
    gdf["piliers_applicables_nb"] = gdf["_est_littorale"].apply(
        lambda x: len(expected) if x else len([p for p in expected if p != "erosion"])
    )
    
    # Scores disponibles
    score_cols = [f"score_{p}" for p in expected]
    for col in score_cols:
        if col not in gdf.columns:
            gdf[col] = np.nan
    
    scores = gdf[score_cols].apply(pd.to_numeric, errors="coerce")
    available = scores.notna()
    
    # Érosion non applicable → ne compte pas comme "disponible"
    if "score_erosion" in scores.columns:
        mask = gdf["_est_littorale"].fillna(False).astype(bool)
        available.loc[~mask, "score_erosion"] = False
    
    gdf["piliers_disponibles_nb"] = available.sum(axis=1)
    gdf["couverture_icv_pct"] = (
        100 * gdf["piliers_disponibles_nb"] / gdf["piliers_applicables_nb"].replace(0, np.nan)
    )
    
    # Calcul ICV pondéré
    weights_series = pd.Series({f"score_{p}": weights[p] for p in expected})
    numerator = scores.where(available, 0).mul(weights_series, axis=1).sum(axis=1)
    denominator = available.mul(weights_series, axis=1).sum(axis=1)
    candidate = numerator / denominator.replace(0, np.nan)
    
    enough = gdf["couverture_icv_pct"] >= min_coverage * 100
    gdf["ICV"] = candidate.where(enough)
    
    # Statut à 3 états (DISPONIBLE / NON_APPLICABLE / INDISPONIBLE)
    gdf["statut_icv"] = np.select(
        [gdf["couverture_icv_pct"].isna() | (gdf["couverture_icv_pct"] < min_coverage * 100),
         gdf["couverture_icv_pct"] < 100],
        ["INDISPONIBLE", "DISPONIBLE_PARTIEL"],
        default="DISPONIBLE",
    )
    
    gdf["classe"] = pd.cut(
        gdf["ICV"],
        bins=[-0.001, 25, 50, 75, 100.001],
        labels=["Faible", "Modérée", "Élevée", "Critique"],
        include_lowest=True,
    )
    
    return gdf.drop(columns=["_est_littorale"], errors="ignore")


def double_peine_scenarios(gdf, seuils=None):
    """
    Calcule la double peine pour plusieurs paires de seuils.
    Retourne un DataFrame de synthèse : seuil x nombre de communes.
    Ne choisit PAS un "meilleur" seuil — l'utilisateur décide.
    """
    seuils = seuils or [(60, 30), (70, 50), (75, 55)]
    resultats = []
    
    for deficit_thr, flood_thr in seuils:
        temp = double_peine(gdf, deficit_threshold=deficit_thr, flood_threshold=flood_thr)
        n_identifiee = int((temp["double_peine_statut"] == "Double peine identifiée").sum())
        n_valides = int((temp["double_peine_statut"] != "Données insuffisantes").sum())
        resultats.append({
            "seuil_services": deficit_thr,
            "seuil_inondation": flood_thr,
            "communes_identifiees": n_identifiee,
            "communes_valides": n_valides,
            "pct_identifiees": round(100 * n_identifiee / max(n_valides, 1), 1),
        })
    return pd.DataFrame(resultats)

def double_peine(gdf, deficit_threshold=60, flood_threshold=30):
    """Double peine : déficit de services ET exposition inondation.
    
    CORRECTION AUDIT-01 : les données manquantes restent INCONNUES.
    Elles ne sont JAMAIS assimilées à zéro.
    
    Note sur flood_threshold : le FRI est sur 0-100 (max observé 64.5).
    Un seuil de 30 correspond à "risque modéré à élevé".
    L'ancien seuil de 55 était trop restrictif (max FRI = 64.5).
    """
    gdf = gdf.copy()

    service_cols = [f"score_{p}" for p in ("sante", "eau", "alimentation")
                    if f"score_{p}" in gdf.columns]

    if service_cols:
        services = gdf[service_cols].apply(pd.to_numeric, errors="coerce")
        gdf["couverture_services_nb"] = services.notna().sum(axis=1)
        gdf["score_services"] = services.mean(axis=1, skipna=True)
        # Exiger au moins 2 piliers documentés sur 3
        gdf.loc[gdf["couverture_services_nb"] < 2, "score_services"] = np.nan
    else:
        gdf["couverture_services_nb"] = 0
        gdf["score_services"] = np.nan

    if "score_inondation" not in gdf.columns:
        gdf["score_inondation"] = np.nan

    gdf["score_inondation"] = pd.to_numeric(gdf["score_inondation"], errors="coerce")

    donnees_valides = gdf["score_services"].notna() & gdf["score_inondation"].notna()

    gdf["double_peine"] = (
        donnees_valides
        & (gdf["score_services"] >= deficit_threshold)
        & (gdf["score_inondation"] >= flood_threshold)
    )

    gdf["double_peine_statut"] = np.select(
        [~donnees_valides, gdf["double_peine"]],
        ["Données insuffisantes", "Double peine identifiée"],
        default="Non identifiée",
    )

    return gdf