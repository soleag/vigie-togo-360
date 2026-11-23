"""
Chargement et harmonisation — Atlas ICV Togo (v5 multi-niveaux & normé).

Stratégie :
  • GPKG FRI = source maîtresse (cantons, préfectures, régions avec noms/pop/FRI)
  • GeoJSON communes + jointure FRI pour les noms
  • Points agrégés par jointure spatiale à chaque niveau
  Ajoute : érosion côtière, enrichissement PCIAEPH, registre des sources.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "essentiel")
DATA_DIR_FALLBACK = os.path.join(os.path.dirname(__file__), "data")

REGIONS = ["Savanes", "Kara", "Centrale", "Plateaux", "Maritime"]
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

PILLARS = ["sante", "eau", "alimentation", "inondation"]

NORMES = {
    "sante":        {"indicateur": "health_fac_per_10000",   "cible": 2.0,  "source": "OMS"},
    "eau":          {"indicateur": "water_pts_per1000",      "cible": 4.0,  "source": "Sphère (1/250 hab.)"},
    "alimentation": {"indicateur": "food_markets_per_10000", "cible": 1.0,  "source": "Approximation"},
    "inondation":   {"indicateur": "flood_fsi_max",          "cible": 60.0, "source": "FRI > 60"},
    "erosion":      {"indicateur": "erosion_epr_mean",       "cible": -1.5, "source": "Seuil critique m/an"},
}

INDICATORS = {
    # ── SANTÉ ────────────────────────────────────────
    "health_pop_per_fac":   {"pillar": "sante", "direction": "negative",
                              "label": "Habitants par formation sanitaire"},
    "health_fac_per_km2":   {"pillar": "sante", "direction": "positive",
                              "label": "Formations sanitaires / 100 km²"},
    # ── EAU ──────────────────────────────────────────
    "water_pts_per1000":    {"pillar": "eau", "direction": "positive",
                              "label": "Points d'eau pour 1 000 hab."},
    "water_pts_per_km2":    {"pillar": "eau", "direction": "positive",
                              "label": "Points d'eau / 100 km²"},
    # ── ALIMENTATION ─────────────────────────────────
    "food_markets_per1000": {"pillar": "alimentation", "direction": "positive",
                              "label": "Marchés pour 1 000 hab."},
    "food_markets_dens":    {"pillar": "alimentation", "direction": "positive",
                              "label": "Marchés / 1 000 km²"},
    # ── INONDATION (FRI = risque, indicateur principal) ─
    "flood_fri_mean":       {"pillar": "inondation", "direction": "negative",
                              "label": "Indice de risque d'inondation FRI (0-100)"},
    "flood_fsi_max":        {"pillar": "inondation", "direction": "negative",
                              "label": "Susceptibilité inondation FSI (max, 0-100)"},
    # ── ÉROSION (intensité uniquement) ───────────────
    "erosion_intensite":    {"pillar": "erosion", "direction": "negative",
                              "label": "Intensité d'érosion (m/an)"},
}

FILES = {
    "geo_communes":    "geo_communes__*.json",
    "geo_cantons_fri": "geo_cantons_fri__*.gpkg",
    "sante":           "sante_formations__*.csv",
    "eau_bornes":      "eau_bornes__*.csv",
    "eau_chateaux":    "eau_chateaux__*.csv",
    "eau_pmh":         "eau_pmh__*.csv",
    "eau_forages":     "eau_forages__*.csv",
    "alim_marches":    "alim_marches__*.csv",
    "edu_ecoles":      "edu_ecoles__*.csv",
    "prix_wfp":        "alim_prix_wfp__*.csv",
    "erosion":         "erosion__*.geojson*",
    "pciaeph":         "pciaeph_eau__*.csv",
    "pciaeph_commerce":"pciaeph_commerce__*.csv",
    "zaaps":           "zaaps__*.csv",
}


def _log(msg: str) -> None:
    print(f"[loader] {msg}", file=sys.stderr)


def _norm(s) -> str:
    """Normalise une clé de jointure (accents, casse, tirets)."""
    if pd.isna(s):
        return ""
    import unicodedata
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for ch in ("'", "'", "-", "_", "/"):
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _find(key: str) -> str | None:
    pattern = FILES.get(key)
    if not pattern:
        return None
    for d in (DATA_DIR, DATA_DIR_FALLBACK):
        matches = _data_matches(d, pattern)
        if matches:
            return str(matches[0])
    return None


def _data_matches(directory: str, pattern: str) -> list[Path]:
    patterns = [pattern]
    if pattern.endswith(".csv"):
        patterns.append(pattern + ".gz")
    return sorted(
        path
        for candidate in patterns
        for path in Path(directory).glob(candidate)
    )


def _pick(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def get_sources_registry() -> dict:
    p = os.path.join(CONFIG_DIR, "sources_registry.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            _log(f"⚠️  registre sources : {e}")
    return {}


# ════════════════════════════════════════════════════════════════
# 1. Niveaux administratifs (via GPKG FRI)
# ════════════════════════════════════════════════════════════════

def _load_levels_from_fri() -> tuple[dict, gpd.GeoDataFrame] | None:
    fri_path = _find("geo_cantons_fri")
    if not fri_path:
        _log("⚠️  GPKG FRI introuvable")
        return None

    _log(f"Chargement FRI : {os.path.basename(fri_path)}")
    fri = gpd.read_file(fri_path).to_crs("EPSG:4326")
    _log(f"  → {len(fri)} cantons bruts")

    col_map = {
        "canton_id":   _pick(fri, ["canton_id", "canton_code"]),
        "canton_nom":  _pick(fri, ["canton_nom", "canton_n_1", "canton_no"]),
        "commune_id":  _pick(fri, ["commune_id"]),
        "commune_nom": _pick(fri, ["commune_no", "commune_nom"]),
        "prefecture":  _pick(fri, ["prefectu_1", "prefecture"]),
        "region_nom":  _pick(fri, ["region_nom", "region"]),
        "pop":         _pick(fri, ["total_pop", "population"]),
        # ✅ CORRECTION AUDIT P0 : FRI et FSI séparés
        "FRI":         _pick(fri, ["FRI"]),
        "norm_fsi":    _pick(fri, ["norm_fsi"]),
        "max_fsi":     _pick(fri, ["max_fsi"]),
    }
    if not col_map["canton_nom"]:
        _log(f"⚠️  colonnes FRI non identifiées : {list(fri.columns)}")
        return None

    _log(f"  Champs détectés : FRI={col_map['FRI']} | norm_fsi={col_map['norm_fsi']} | max_fsi={col_map['max_fsi']}")

    # ✅ FRI (risque, 0-1 → 0-100)
    if col_map["FRI"]:
        fri["_fri_100"] = pd.to_numeric(fri[col_map["FRI"]], errors="coerce") * 100
    else:
        fri["_fri_100"] = np.nan

    # ✅ FSI normalisé (susceptibilité, 0-1 → 0-100). Fallback sur max_fsi si absent.
    if col_map["norm_fsi"]:
        fri["_fsi_100"] = pd.to_numeric(fri[col_map["norm_fsi"]], errors="coerce") * 100
    elif col_map["max_fsi"]:
        mx = pd.to_numeric(fri[col_map["max_fsi"]], errors="coerce").max()
        if pd.notna(mx) and mx > 0:
            fri["_fsi_100"] = (pd.to_numeric(fri[col_map["max_fsi"]], errors="coerce") / mx) * 100
        else:
            fri["_fsi_100"] = np.nan
    else:
        fri["_fsi_100"] = np.nan

    levels = {}

    # ═══════════════════════════════════════════════════════════
    # CANTONS
    # ═══════════════════════════════════════════════════════════
    cantons = fri.copy()
    cantons["id"] = cantons[col_map["canton_id"]].astype(str) if col_map["canton_id"] else range(len(cantons))
    cantons["commune"] = cantons[col_map["canton_nom"]]
    cantons["region"] = cantons[col_map["region_nom"]] if col_map["region_nom"] else "N/D"
    cantons["prefecture"] = cantons[col_map["prefecture"]] if col_map["prefecture"] else "N/D"
    cantons["population"] = cantons[col_map["pop"]] if col_map["pop"] else np.nan
    cantons["flood_fri_mean"] = cantons["_fri_100"]
    cantons["flood_fsi_max"] = cantons["_fsi_100"]
    cantons["area_km2"] = cantons.to_crs("EPSG:32631").geometry.area / 1e6
    levels["cantons"] = cantons[["id", "commune", "region", "prefecture",
                                  "population", "flood_fri_mean", "flood_fsi_max",
                                  "area_km2", "geometry"]].copy()

    # ═══════════════════════════════════════════════════════════
    # PRÉFECTURES (dissolve)
    # ═══════════════════════════════════════════════════════════
    if col_map["prefecture"]:
        pref_agg = {"_fri_100": "mean", "_fsi_100": "max"}
        if col_map["pop"]: pref_agg[col_map["pop"]] = "sum"
        if col_map["region_nom"]: pref_agg[col_map["region_nom"]] = "first"
        prefs = fri.dissolve(by=col_map["prefecture"], aggfunc=pref_agg).reset_index()
        prefs = prefs.rename(columns={col_map["prefecture"]: "id"})
        prefs["commune"] = prefs["id"]
        prefs["region"] = prefs[col_map["region_nom"]] if col_map["region_nom"] else "N/D"
        prefs["prefecture"] = prefs["id"]
        prefs["population"] = prefs[col_map["pop"]] if col_map["pop"] else np.nan
        prefs["flood_fri_mean"] = prefs["_fri_100"]
        prefs["flood_fsi_max"] = prefs["_fsi_100"]
        prefs["area_km2"] = prefs.to_crs("EPSG:32631").geometry.area / 1e6
        levels["prefectures"] = prefs[["id", "commune", "region", "prefecture",
                                        "population", "flood_fri_mean", "flood_fsi_max",
                                        "area_km2", "geometry"]].copy()

    # ═══════════════════════════════════════════════════════════
    # RÉGIONS (dissolve)
    # ═══════════════════════════════════════════════════════════
    if col_map["region_nom"]:
        reg_agg = {"_fri_100": "mean", "_fsi_100": "max"}
        if col_map["pop"]: reg_agg[col_map["pop"]] = "sum"
        regs = fri.dissolve(by=col_map["region_nom"], aggfunc=reg_agg).reset_index()
        regs = regs.rename(columns={col_map["region_nom"]: "commune"})
        regs["id"] = regs["commune"]
        regs["region"] = regs["commune"]
        regs["prefecture"] = "N/D"
        regs["population"] = regs[col_map["pop"]] if col_map["pop"] else np.nan
        regs["flood_fri_mean"] = regs["_fri_100"]
        regs["flood_fsi_max"] = regs["_fsi_100"]
        regs["area_km2"] = regs.to_crs("EPSG:32631").geometry.area / 1e6
        levels["regions"] = regs[["id", "commune", "region", "prefecture",
                                   "population", "flood_fri_mean", "flood_fsi_max",
                                   "area_km2", "geometry"]].copy()

    for k, g in levels.items():
        _log(f"  → {len(g)} {k}")
    return levels, fri


# ════════════════════════════════════════════════════════════════
# 2. Communes
# ════════════════════════════════════════════════════════════════

def _load_communes(fri: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    geo_path = _find("geo_communes")
    if not geo_path:
        return None
    _log(f"Chargement communes : {os.path.basename(geo_path)}")
    gdf = gpd.read_file(geo_path).to_crs("EPSG:4326")
    gdf["commune_id"] = gdf["id"].astype(str).str.replace("communes.", "", regex=False)
    _log(f"  → {len(gdf)} polygones")

    if fri is not None:
        cid_fri = _pick(fri, ["commune_id"])
        cname   = _pick(fri, ["commune_no", "commune_nom"])
        region  = _pick(fri, ["region_nom", "region"])
        pref    = _pick(fri, ["prefectu_1", "prefecture"])
        pop     = _pick(fri, ["total_pop", "population"])

        if cid_fri and cname:
            agg = {"commune": (cname, "first")}
            if region: agg["region"]     = (region, "first")
            if pref:   agg["prefecture"] = (pref, "first")
            if pop:    agg["population"] = (pop, "sum")
            # ✅ CORRECTION : FRI et FSI séparés
            if "_fri_100" in fri.columns: agg["flood_fri_mean"] = ("_fri_100", "mean")
            if "_fsi_100" in fri.columns: agg["flood_fsi_max"]  = ("_fsi_100", "max")

            comm_tab = fri.groupby(cid_fri).agg(**agg).reset_index()
            comm_tab = comm_tab.rename(columns={cid_fri: "commune_id"})
            gdf = gdf.merge(comm_tab, on="commune_id", how="left")
            _log(f"  → jointure FRI : {gdf['commune'].notna().sum()}/{len(gdf)}")

    # Garantir toutes les colonnes
    for col, default in [("commune", None), ("region", "N/D"),
                         ("prefecture", "N/D"), ("population", np.nan),
                         ("flood_fri_mean", np.nan), ("flood_fsi_max", np.nan)]:
        if col not in gdf.columns:
            gdf[col] = gdf["commune_id"] if col == "commune" else default
    gdf["commune"] = gdf["commune"].fillna(gdf["commune_id"])
    gdf["commune"] = gdf["commune"].astype("string").str.strip()
    gdf["commune"] = gdf["commune"].replace("", pd.NA).fillna(gdf["commune_id"])
    gdf["region"] = gdf["region"].fillna("N/D").replace("", "N/D")
    gdf["prefecture"] = gdf["prefecture"].fillna("N/D").replace("", "N/D")

    gdf["id"] = gdf["commune_id"]
    gdf["area_km2"] = gdf.to_crs("EPSG:32631").geometry.area / 1e6
    return gdf[["id", "commune", "region", "prefecture",
                "population", "flood_fri_mean", "flood_fsi_max",
                "area_km2", "geometry"]].copy()

# ════════════════════════════════════════════════════════════════
# 3. Points (santé, eau, marchés, écoles, érosion)
# ════════════════════════════════════════════════════════════════

def _load_points() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Retourne (points_simples, points_avec_attributs)."""
    simples, attrition = [], []

    sources = {
        "sante":   ["sante"],
        "eau":     ["eau_bornes", "eau_chateaux", "eau_pmh", "eau_forages"],
        "marches": ["alim_marches"],
        "ecoles":  ["edu_ecoles"],
    }
    for ptype, keys in sources.items():
        for k in keys:
            p = _find(k)
            if not p:
                continue
            try:
                df = pd.read_csv(p, low_memory=False)
                if "geometry" not in df.columns:
                    continue
                g = gpd.GeoDataFrame(
                    df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
                    crs="EPSG:4326",
                )
                # Nettoyage : garder seulement les géométries valides
                g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
                g = g[["geometry"]].copy()
                g["type"] = ptype
                if len(g) > 0:
                    simples.append(g)
            except Exception as e:
                _log(f"  ⚠️  {os.path.basename(p)} : {e}")

    # Érosion : transects → centroïdes (avec nettoyage strict)
    p = _find("erosion")
    if p:
        try:
            ero = gpd.read_file(p).to_crs("EPSG:3857")
            # Calcul des centroïdes en projection métrique
            centroids = ero.geometry.centroid
            ero = ero.copy()
            ero["geometry"] = centroids
            ero = ero.to_crs("EPSG:4326")

            # Nettoyage : écarter les centroïdes vides ou invalides
            mask = (ero.geometry.notna()
                    & ~ero.geometry.is_empty
                    & ero.geometry.is_valid)
            ero = ero[mask].copy()

            ero["type"] = "erosion"
            cols_keep = ["geometry", "type"]
            for c in ("EPR", "LRR", "NSM"):
                if c in ero.columns:
                    ero[c] = pd.to_numeric(ero[c], errors="coerce")
                    cols_keep.append(c)

            if len(ero) > 0:
                attrition.append(ero[cols_keep])
                _log(f"  → {len(ero)} transects d'érosion (centroïdes valides)")
            else:
                _log("  ⚠️  érosion : aucun centroïde valide")
        except Exception as e:
            _log(f"  ⚠️  érosion : {e}")

    # ── Helper robuste de concaténation ────────────────────────
    def _to_gdf(lst):
        if not lst:
            return gpd.GeoDataFrame()
        try:
            merged = pd.concat(lst, ignore_index=True)
            return gpd.GeoDataFrame(merged, geometry="geometry",
                                    crs="EPSG:4326")
        except Exception as e:
            _log(f"  ⚠️  concat GeoDataFrame : {e}")
            return gpd.GeoDataFrame()

    pts_simple = _to_gdf(simples)
    pts_attr = _to_gdf(attrition)

    total = len(pts_simple) + len(pts_attr)
    _log(f"  → {total} points géolocalisés au total "
         f"({len(pts_simple)} simples, {len(pts_attr)} avec attributs)")
    return pts_simple, pts_attr


def _aggregate_points(level_gdf, pts_simple, pts_attr):
    # Comptages simples
    if not pts_simple.empty:
        joined = gpd.sjoin(pts_simple, level_gdf[["id", "geometry"]],
                           how="inner", predicate="within")
        counts = joined.groupby(["id", "type"]).size().unstack(fill_value=0)
        counts = counts.rename(columns={
            "sante": "nb_sante", "eau": "nb_eau",
            "marches": "nb_marches", "ecoles": "nb_ecoles",
        })
        level_gdf = level_gdf.merge(counts, left_on="id", right_index=True, how="left")

    for col in ("nb_sante", "nb_eau", "nb_marches", "nb_ecoles"):
        if col not in level_gdf.columns:
            level_gdf[col] = 0
        level_gdf[col] = level_gdf[col].fillna(0).astype(int)

    # Érosion : moyennes/max
    if not pts_attr.empty:
        joined = gpd.sjoin(pts_attr, level_gdf[["id", "geometry"]],
                           how="inner", predicate="within")
        ero = joined[joined["type"] == "erosion"]
        if len(ero) > 0:
            ero = ero.copy()
            for col in ("EPR", "LRR", "NSM"):
                if col in ero.columns:
                    ero[col] = pd.to_numeric(ero[col], errors="coerce")
                    ero.loc[ero[col].eq(9999.99), col] = np.nan
            agg = {"id": "first"}
            if "EPR" in ero.columns:
                agg["erosion_epr_mean"] = ("EPR", "mean")
                agg["erosion_epr_max_abs"] = ("EPR", lambda s: s.abs().max())
            if "LRR" in ero.columns:
                agg["erosion_lrr_mean"] = ("LRR", "mean")
            # groupby par id
            ero_agg = ero.groupby("id").agg(
                **{k: v for k, v in agg.items() if k != "id"}
            ).reset_index()
            level_gdf = level_gdf.merge(ero_agg, on="id", how="left")

    for col in ("erosion_epr_mean", "erosion_epr_max_abs", "erosion_lrr_mean"):
        if col not in level_gdf.columns:
            level_gdf[col] = np.nan
    return level_gdf


# ════════════════════════════════════════════════════════════════
# 4. PCIAEPH (enrichissement eau)
# ════════════════════════════════════════════════════════════════

def _load_pciaeph(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Charge le fichier PCIAEPH eau (colonnes profondeur, débit)."""
    p = _find("pciaeph")
    if not p:
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        _log(f"  ⚠️  PCIAEPH : lecture impossible ({e})")
        return pd.DataFrame()

    _log(f"PCIAEPH : {len(df)} sous-projets")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    sub = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(sub) == 0:
        return pd.DataFrame()

    try:
        pts = gpd.GeoDataFrame(
            sub, geometry=gpd.points_from_xy(sub["longitude"], sub["latitude"]),
            crs="EPSG:4326",
        )
        comm = communes[["id", "geometry"]].rename(columns={"id": "_commune_id"})
        joined = gpd.sjoin(pts, comm, how="inner", predicate="within")
        if "_commune_id" not in joined.columns:
            return pd.DataFrame()

        # Uniquement les colonnes propres à _load_pciaeph (avancement et coût
        # viennent de _load_pciaeph_detailed)
        agg = {}
        if "depth_of_drilling" in joined.columns:
            agg["pciaeph_profondeur"] = ("depth_of_drilling", "mean")
        if "drilling_flow_rate" in joined.columns:
            agg["pciaeph_debit"] = ("drilling_flow_rate", "mean")

        if not agg:
            _log("  ⚠️  PCIAEPH : aucune colonne exploitable")
            return pd.DataFrame()

        result = (joined.groupby("_commune_id").agg(**agg)
                       .reset_index()
                       .rename(columns={"_commune_id": "id"}))
        _log(f"  → {len(result)} communes enrichies (sur {len(sub)} sous-projets)")
        return result
    except Exception as e:
        _log(f"  ⚠️  PCIAEPH : erreur agrégation ({e})")
        return pd.DataFrame()
        
def _check_pciaeph_coherence() -> list[str]:
    """Détecte l'incohérence population vs hommes+femmes dans PCIAEPH.
    Retourne une liste vide si les colonnes sont absentes (fichier commerce, etc.)."""
    alertes = []
    p = _find("pciaeph")
    if not p:
        return alertes
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return alertes

    need = ["population", "direct_beneficiaries_men",
            "direct_beneficiaries_women"]
    present = [c for c in need if c in df.columns]
    if len(present) < 3:
        # Fichier sans les 3 colonnes → on skip silencieusement
        _log(f"  ℹ️  Audit PCIAEPH : colonnes manquantes "
             f"({set(need) - set(present)}) → audit ignoré")
        return alertes

    try:
        pop = pd.to_numeric(df["population"], errors="coerce")
        hf = (pd.to_numeric(df["direct_beneficiaries_men"], errors="coerce").fillna(0)
              + pd.to_numeric(df["direct_beneficiaries_women"], errors="coerce").fillna(0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ecart = (hf - pop).abs() / pop.replace(0, np.nan) * 100
        n_incoherent = int((ecart > 10).sum())
        n_valides = int(pop.notna().sum())
        if n_incoherent > 0:
            alertes.append(
                f"⚠️ PCIAEPH : {n_incoherent}/{n_valides} sous-projets ont "
                f"une incohérence population ≠ hommes+femmes")
        else:
            alertes.append(
                f"✅ PCIAEPH : {n_valides}/{len(df)} sous-projets cohérents")
    except Exception as e:
        _log(f"  ⚠️  Audit PCIAEPH échoué : {e}")
    return alertes    

# ════════════════════════════════════════════════════════════════
# 4bis. Projets détaillés et agriculture
# ════════════════════════════════════════════════════════════════

def _classify_statut(s) -> str:
    """Classe un statut PCIAEPH libre en catégorie normalisée."""
    if pd.isna(s):
        return "inconnu"
    t = str(s).lower().strip()
    if "réception" in t or "reception" in t:
        if "définitif" in t or "definitif" in t:
            return "receptionne_def"
        return "receptionne_prov"
    if any(k in t for k in ("achevé", "acheve", "terminé", "termine",
                             "complete", "completed", "finish")):
        return "acheve"
    if any(k in t for k in ("cours", "progress", "ongoing", "construction")):
        return "en_cours"
    if any(k in t for k in ("planifi", "planned", "préparat", "preparat")):
        return "planifie"
    return "autre"


def _load_pciaeph_detailed(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extrait les statistiques détaillées des projets COSO par commune."""
    p = _find("pciaeph")
    if not p:
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        _log(f"  ⚠️  PCIAEPH détaillé : {e}")
        return pd.DataFrame()

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    sub = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(sub) == 0:
        return pd.DataFrame()

    pts = gpd.GeoDataFrame(
        sub,
        geometry=gpd.points_from_xy(sub["longitude"], sub["latitude"]),
        crs="EPSG:4326",
    )
    comm = communes[["id", "geometry"]].rename(columns={"id": "_cid"})
    joined = gpd.sjoin(pts, comm, how="inner", predicate="within")

    if "_cid" not in joined.columns:
        return pd.DataFrame()

    # Statut principal
    statut_col = None
    for c in ("current_status_of_the_site", "status"):
        if c in joined.columns:
            statut_col = c
            break
    joined["_statut"] = (joined[statut_col].map(_classify_statut)
                         if statut_col else "inconnu")

    # Colonnes numériques
    for c, alias in [("progress_percent", "_progress"),
                     ("estimated_cost", "_cout"),
                     ("total_contract_amount_paid", "_paye"),
                     ("population", "_pop"),
                     ("direct_beneficiaries_men", "_benef_h"),
                     ("direct_beneficiaries_women", "_benef_f")]:
        if c in joined.columns:
            joined[alias] = pd.to_numeric(joined[c], errors="coerce")

    # Agrégation
        # Agrégation
    agg_spec = {"pciaeph_total": ("_statut", "size")}
    for st in ("en_cours", "acheve", "receptionne_prov",
               "receptionne_def", "planifie"):
        agg_spec[f"pciaeph_{st}"] = ("_statut", lambda x, s=st: int((x == s).sum()))
    for alias, name in [("_progress", "pciaeph_avancement"),
                        ("_cout", "pciaeph_cout_total"),
                        ("_paye", "pciaeph_cout_paye"),
                        ("_pop", "pciaeph_pop_cible"),
                        ("_benef_h", "pciaeph_benef_hommes"),
                        ("_benef_f", "pciaeph_benef_femmes")]:
        if alias in joined.columns:
            agg_spec[name] = (alias, "mean" if alias == "_progress" else "sum")

    result = joined.groupby("_cid").agg(**agg_spec).reset_index()
    result = result.rename(columns={"_cid": "id"})

    # Dérivés — protégés par des "if"
    if "pciaeph_total" in result.columns:
        receptionnes = (result.get("pciaeph_acheve", 0)
                       + result.get("pciaeph_receptionne_prov", 0)
                       + result.get("pciaeph_receptionne_def", 0))
        result["pciaeph_taux_reception"] = (
            receptionnes / result["pciaeph_total"].replace(0, np.nan) * 100
        )
    if "pciaeph_cout_total" in result.columns and "pciaeph_cout_paye" in result.columns:
        result["pciaeph_taux_decaissement"] = (
            result["pciaeph_cout_paye"]
            / result["pciaeph_cout_total"].replace(0, np.nan) * 100
        )
    _log(f"  → PCIAEPH détaillé : {len(result)} communes "
         f"({[c for c in result.columns if c.startswith('pciaeph_')][:3]}…)")
    return result


def _load_zaaps(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Charge les ZAAPs et calcule surface/nombre par commune."""
    p = _find("zaaps")
    if not p:
        return pd.DataFrame()

    try:
        if p.endswith((".geojson", ".json")):
            gdf = gpd.read_file(p)
        else:
            df = pd.read_csv(p, low_memory=False)
            if "geometry" in df.columns:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
                    crs="EPSG:4326",
                )
            else:
                return pd.DataFrame()

        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
            return pd.DataFrame()

        gdf = gdf.to_crs("EPSG:4326")

        # Compter par centroïde (pour éviter les doubles comptages)
        # Compter par centroïde (reprojeter pour précision)
        gdf_m = gdf.to_crs("EPSG:3857")
        centroids = gdf_m.geometry.centroid.to_crs("EPSG:4326")
        pts = gpd.GeoDataFrame(
            gdf.drop(columns="geometry", errors="ignore").reset_index(drop=True),
            geometry=centroids.reset_index(drop=True),
            crs="EPSG:4326",
        )

        comm = communes[["id", "geometry"]].rename(columns={"id": "_cid"})
        joined = gpd.sjoin(pts, comm, how="inner", predicate="within")

        if "_cid" not in joined.columns:
            return pd.DataFrame()

        result = joined.groupby("_cid").size().reset_index(name="zaap_nb")
        result = result.rename(columns={"_cid": "id"})

        # Surface totale en ha
        try:
            gdf_m = gdf.to_crs("EPSG:3857")
            comm_m = communes[["id", "geometry"]].to_crs("EPSG:3857")
            inter = gpd.overlay(
                gdf_m[["geometry"]].reset_index(drop=True),
                comm_m.rename(columns={"id": "_cid"}),
                how="intersection",
                keep_geom_type=True,
            )
            inter["_ha"] = inter.geometry.area / 10000
            surf = inter.groupby("_cid")["_ha"].sum().reset_index()
            surf = surf.rename(columns={"_cid": "id", "_ha": "zaap_surface_ha"})
            result = result.merge(surf, on="id", how="left")
        except Exception:
            result["zaap_surface_ha"] = np.nan

        _log(f"  → ZAAPs : {len(result)} communes, "
             f"{int(result['zaap_nb'].sum())} zones")
        return result
    except Exception as e:
        _log(f"  ⚠️  ZAAPs : {e}")
        return pd.DataFrame()
        
def _load_all_projects(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Catalogue unifié de TOUS les sous-projets COSO (multi-secteurs)."""
    candidates = []
    for d in (DATA_DIR,):
        if not Path(d).exists():
            continue
        for pat in ("pciaeph_*.csv",           # Anciens noms
                    "coso_*.csv",              # Nouveaux noms (coso_agriculture__...)
                    "*subproject*sector*.csv*",
                    "*coso*.csv*"): candidates.extend(_data_matches(d, pat))

    seen, files = set(), []
    for f in candidates:
        if f.name not in seen:
            seen.add(f.name)
            files.append(f)

    if not files:
        _log("⚠️  Aucun fichier COSO trouvé")
        return pd.DataFrame()

    _log(f"Catalogue projets : {len(files)} fichier(s) COSO détecté(s)")
    rows = []

    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            _log(f"  ⚠️  {path.name} : {e}")
            continue
        if len(df) == 0:
            continue

        # Secteur
        fname = path.name.lower()
        if "eau" in fname or "hydraulique" in fname:
            secteur = "Eau & hydraulique"
        elif "commerce" in fname or "marche" in fname:
            secteur = "Commerce & marchés"
        elif "sant" in fname:
            secteur = "Santé"
        elif "educ" in fname:
            secteur = "Éducation"
        else:
            secteur = "Autre"

        def col(*cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None

        c_id     = col("id", "number")
        c_title  = col("title", "name", "nom")
        c_status = col("current_status_of_the_site", "status")
        c_prog   = col("progress_percent")
        c_cost   = col("estimated_cost")
        c_paid   = col("total_contract_amount_paid")
        c_lat    = col("latitude")
        c_lon    = col("longitude")
        c_comm   = col("commune_nom_bdd", "commune", "location_name")
        c_region = col("region_nom_bdd", "region")

        for _, r in df.iterrows():
            rows.append({
                "project_id":       r.get(c_id, np.nan) if c_id else np.nan,
                "titre":            str(r.get(c_title, ""))[:120] if c_title else "—",
                "secteur":          secteur,
                "statut_brut":      str(r.get(c_status, ""))[:80] if c_status else "",
                "avancement_pct":   pd.to_numeric(r.get(c_prog), errors="coerce"),
                "cout_total":       pd.to_numeric(r.get(c_cost), errors="coerce"),
                "cout_paye":        pd.to_numeric(r.get(c_paid), errors="coerce"),
                "region":           str(r.get(c_region, "")).strip(),
                "commune_nom":      str(r.get(c_comm, "")).strip(),
                "latitude":         pd.to_numeric(r.get(c_lat), errors="coerce"),
                "longitude":        pd.to_numeric(r.get(c_lon), errors="coerce"),
                "source_fichier":   path.name,
            })

    if not rows:
        return pd.DataFrame()

    cat = pd.DataFrame(rows)

    def norm_statut(s):
        if pd.isna(s): return "Inconnu"
        t = str(s).lower()
        if "réception" in t or "reception" in t:
            return "Réceptionné définitif" if "définitif" in t else "Réceptionné provisoire"
        if any(k in t for k in ("achevé", "acheve", "terminé", "termine", "complete")):
            return "Achevé"
        if any(k in t for k in ("cours", "progress", "ongoing")):
            return "En cours"
        if any(k in t for k in ("planifi", "planned", "préparat")):
            return "Planifié"
        if any(k in t for k in ("arrêt", "suspend", "abandon")):
            return "Suspendu"
        return "Autre"

    cat["statut"] = cat["statut_brut"].map(norm_statut)

    # Jointure spatiale
    if communes is not None and not communes.empty:
        try:
            valid = cat.dropna(subset=["latitude", "longitude"]).copy()
            if len(valid) > 0:
                pts = gpd.GeoDataFrame(
                    valid,
                    geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
                    crs="EPSG:4326",
                )
                cr = communes[["commune", "region", "geometry"]].rename(
                    columns={"commune": "commune_officielle",
                             "region": "region_officielle"})
                j = gpd.sjoin(pts, cr, how="left", predicate="within")
                j = j.drop(columns=["geometry", "index_right"], errors="ignore")
                j = j.drop_duplicates(subset=["project_id"])
                cat = cat.merge(j[["project_id", "commune_officielle",
                                   "region_officielle"]],
                                on="project_id", how="left")
                cat["commune"] = cat["commune_officielle"].fillna(cat["commune_nom"])
                cat["region"] = cat["region_officielle"].fillna(cat["region"])
            else:
                cat["commune"] = cat["commune_nom"]
        except Exception as e:
            _log(f"  ⚠️  jointure spatiale : {e}")
            cat["commune"] = cat["commune_nom"]
    else:
        cat["commune"] = cat["commune_nom"]

    cat["commune"] = cat["commune"].replace("", "Non localisé").fillna("Non localisé")
    cat["region"] = cat["region"].replace("", "N/D").fillna("N/D")

    _log(f"  → {len(cat)} projets unifiés sur {cat['secteur'].nunique()} secteur(s)")
    for s, n in cat["secteur"].value_counts().items():
        _log(f"     • {s} : {n}")

    return cat


# ════════════════════════════════════════════════════════════════
# 4quater. ÉDUCATION — 3 défis stratégiques
# ════════════════════════════════════════════════════════════════


def _load_education_indicators() -> pd.DataFrame:
    """Charge les indicateurs éducation régionaux.
    Filtre sexe='Total' et exclut les agrégats 'TOGO'.
    Utilise la valeur la plus récente par région."""
    indicators = {}
    files_map = {
        "abandon":      ["edu_abandon__*.csv", "taux-dabandon*.csv"],
        "achevement":   ["edu_achevement__*.csv", "taux-dachevement*.csv"],
        "redoublement": ["edu_redoublement__*.csv", "taux-de-redoublement*.csv"],
        "scol_prim":    ["edu_scolarisation__primaire.csv", "taux-brut-de-scolarisation-au-primaire*.csv"],
        "scol_lycee":   ["edu_scolarisation__lycee.csv", "taux-brut-de-scolarisation-au-lycee*.csv"],
    }
    _REGION_MAP = {
        "savanes": "Savanes", "kara": "Kara", "centrale": "Centrale",
        "plateaux": "Plateaux", "maritime": "Maritime",
    }

    for key, patterns in files_map.items():
        p_file = None
        for pat in patterns:
            for d in (DATA_DIR, DATA_DIR_FALLBACK):
                if not Path(d).exists():
                    continue
                matches_f = _data_matches(d, pat)
                if matches_f:
                    p_file = matches_f[0]
                    break
            if p_file:
                break

        if not p_file:
            _log(f"  ⚠️  Éducation / {key} : fichier introuvable")
            continue

        try:
            df = pd.read_csv(p_file, low_memory=False)
        except Exception as e:
            _log(f"  ⚠️  {p_file.name} : {e}")
            continue

        # Détecter la colonne région (gère accents)
        region_col = next((c for c in df.columns
                          if "région" in c.lower() or "region" in c.lower()), None)
        if not region_col:
            _log(f"  ⚠️  {p_file.name} : pas de colonne région")
            continue

        # Détecter Date et Value
        date_col = next((c for c in df.columns
                        if c.lower() in ("date", "year", "année", "annee")), None)
        value_col = next((c for c in df.columns
                         if c.lower() in ("value", "valeur", "taux", "rate")), None)

        if not value_col:
            _log(f"  ⚠️  {p_file.name} : pas de colonne Value")
            continue

        # Filtrer : sexe = Total (évite doublons)
        sexe_col = next((c for c in df.columns
                        if c.lower() in ("sexe", "sex", "genre")), None)
        if sexe_col:
            df = df[df[sexe_col].astype(str).str.lower().isin(
                ["total", "ensemble", "both", "tous"])]

        # Normaliser région et filtrer les 5 vraies régions
        df["_region_norm"] = df[region_col].astype(str).str.lower().str.strip()
        df = df[df["_region_norm"].isin(_REGION_MAP.keys())].copy()
        df["region"] = df["_region_norm"].map(_REGION_MAP)
        df[key] = pd.to_numeric(df[value_col], errors="coerce")

        if date_col:
            df["_date"] = pd.to_numeric(df[date_col], errors="coerce")
            # Garder la valeur la plus récente par région
            df = df.sort_values("_date").groupby("region").tail(1)

        # Convertir ratio → pourcentage si nécessaire
        if df[key].dropna().any() and df[key].dropna().max() <= 1.5:
            df[key] = df[key] * 100

        agg = df.groupby("region")[key].mean().to_dict()
        if agg:
            indicators[key] = agg
            _log(f"  ✓ Éducation / {key} : {len(agg)} régions "
                 f"(région={region_col}, valeur={value_col})")
        else:
            _log(f"  ⚠️  {p_file.name} : aucune région reconnue")

    if not indicators:
        _log("  ⚠️  Aucun fichier éducation exploitable")
        return pd.DataFrame()

    result = pd.DataFrame(indicators)
    result.index.name = "region"
    return result.reset_index()


def _load_schools_by_type(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Charge tous les types d'écoles (maternelle, primaire, collège, lycée)
    et compte par commune."""
    types_files = {
        "maternelle": "edu_ecoles__maternelle.csv",
        "primaire":   "edu_ecoles__primaire.csv",
        "college":    "edu_ecoles__college.csv",
        "lycee":      "edu_ecoles__lycee.csv",
    }

    counts_by_type = {}
    for type_name, pattern in types_files.items():
        p = _find_any(pattern) if "_find_any" in globals() else None
        if not p:
            for d in (DATA_DIR, DATA_DIR_FALLBACK):
                if not Path(d).exists():
                    continue
                matches = _data_matches(d, pattern)
                if matches:
                    p = matches[0]
                    break

        if not p:
            _log(f"  ⚠️  Écoles / {type_name} : {pattern} introuvable")
            continue

        try:
            df = pd.read_csv(p, low_memory=False)
            key_col = next((c for c in df.columns
                           if "commune" in c.lower() and "nom" in c.lower()), None)
            if not key_col:
                _log(f"  ⚠️  {p.name} : pas de colonne commune")
                continue

            # Normaliser la clé
            # Normalisation locale (sans dépendance externe)
            import unicodedata
            def _norm_local(s):
                if pd.isna(s):
                    return ""
                s = str(s).strip().lower()
                s = unicodedata.normalize("NFD", s)
                s = "".join(c for c in s if unicodedata.category(c) != "Mn")
                for ch in ("'", "'", "-", "_", "/"):
                    s = s.replace(ch, " ")
                return " ".join(s.split())
            df["_key"] = df[key_col].astype(str).map(_norm_local)
            df = df[df["_key"] != ""]
            counts = df.groupby("_key").size().reset_index(name=f"nb_{type_name}")
            counts_by_type[type_name] = counts
            _log(f"  ✓ Écoles / {type_name} : {len(df)} établissements")
        except Exception as e:
            _log(f"  ⚠️  {type_name} : {e}")

    if not counts_by_type:
        return pd.DataFrame()

    # Fusion progressive
    result = None
    for type_name, counts in counts_by_type.items():
        if result is None:
            result = counts
        else:
            result = result.merge(counts, on="_key", how="outer")

    # Remplir les NaN par 0
    for type_name in types_files:
        col = f"nb_{type_name}"
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    result = result.rename(columns={"_key": "commune_key"})
    return result


def _load_teachers_by_region() -> pd.DataFrame:
    """Charge les effectifs d'enseignants par région/ordre."""
    for pat in ("edu_enseignants__*.csv", "etablissements-scolaires-professeurs.csv",
                "repartition-des-eleves-et-enseignants*.csv"):
        p = None
        for d in (DATA_DIR, DATA_DIR_FALLBACK):
            if not Path(d).exists():
                continue
            matches = _data_matches(d, pat)
            if matches:
                p = matches[0]
                break
        if not p:
            continue

        try:
            df = pd.read_csv(p, low_memory=False)
            region_col = next((c for c in df.columns
                              if "region" in c.lower()), None)
            ens_col = next((c for c in df.columns
                           if any(k in c.lower() for k in ("enseignant", "professeur", "teacher"))), None)

            if not region_col or not ens_col:
                continue

            df["_reg"] = df[region_col].astype(str).str.strip().str.title()
            df["_ens"] = pd.to_numeric(df[ens_col], errors="coerce")
            df = df.dropna(subset=["_ens"])
            df = df[df["_reg"].isin(REGIONS)]

            agg = df.groupby("_reg")["_ens"].sum().reset_index()
            agg.columns = ["region", "nb_enseignants"]
            _log(f"  ✓ Enseignants : {len(agg)} régions")
            return agg
        except Exception as e:
            _log(f"  ⚠️  Enseignants / {p.name} : {e}")
    return pd.DataFrame()

# ════════════════════════════════════════════════════════════════
# 4quinquies. ALIMENTATION — Bilan vivrier + IPC
# ════════════════════════════════════════════════════════════════

def _load_bilan_vivrier() -> pd.DataFrame:
    """Charge le bilan vivrier par région (production par culture).
    Retourne un DataFrame avec région, culture, année, production."""
    p = None
    for d in (DATA_DIR, DATA_DIR_FALLBACK):
        if not Path(d).exists():
            continue
        matches = _data_matches(d, "alim_bilan__region.csv")
        if matches:
            p = matches[0]
            break

    if not p:
        _log("  ⚠️  Bilan vivrier régional introuvable")
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        _log(f"  ⚠️  Bilan vivrier : {e}")
        return pd.DataFrame()

    # Détecter les colonnes
    region_col = next((c for c in df.columns
                      if c.lower() in ("région", "region")), None)
    produit_col = next((c for c in df.columns
                       if "type-de-produit" in c.lower() or
                          "type-de-produits" in c.lower() or
                          "culture" in c.lower()), None)
    libelle_col = next((c for c in df.columns
                       if c.lower() in ("libellés", "libelles", "libelle")), None)
    date_col = next((c for c in df.columns
                    if c.lower() in ("date", "year", "annee", "année")), None)
    value_col = next((c for c in df.columns
                     if c.lower() in ("value", "valeur")), None)

    if not all([region_col, produit_col, date_col, value_col]):
        _log(f"  ⚠️  Bilan vivrier : colonnes manquantes "
             f"(region={region_col}, produit={produit_col}, "
             f"date={date_col}, value={value_col})")
        return pd.DataFrame()

    df = df.rename(columns={
        region_col: "region",
        produit_col: "culture",
        date_col: "annee",
        value_col: "production",
    })

    # Normaliser région
    df["region"] = df["region"].astype(str).str.strip().str.title()
    df = df[df["region"].isin(REGIONS)].copy()

    # Normaliser culture
    df["culture"] = df["culture"].astype(str).str.strip()

    # Filtrer uniquement les cultures individuelles (pas les agrégats)
    cultures_agg = ("CEREALES", "TUBERCULES", "LEGUMINEUSES")
    df = df[~df["culture"].isin(cultures_agg)].copy()

    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["production", "annee"])

    # Garder uniquement la ligne "Production brutes" si plusieurs libellés
    if libelle_col:
        df = df[df[libelle_col].astype(str).str.lower().str.contains("brute", na=False)]

    _log(f"  ✓ Bilan vivrier : {len(df)} enregistrements, "
         f"{df['culture'].nunique()} cultures, {df['region'].nunique()} régions, "
         f"années {int(df['annee'].min())}-{int(df['annee'].max())}")

    return df[["region", "culture", "annee", "production"]].reset_index(drop=True)


def _load_ipc() -> pd.DataFrame:
    """Charge les données IPC (sécurité alimentaire par zone).
    Retourne un DataFrame avec zone, phase, population, pourcentage."""
    p = None
    for d in (DATA_DIR, DATA_DIR_FALLBACK):
        if not Path(d).exists():
            continue
        # Chercher le fichier IPC area
        for pattern in ("alim_ipc__area.csv", "alim_ipc__*.csv"):
            matches = _data_matches(d, pattern)
            if matches:
                p = matches[0]
                break
        if p:
            break

    if not p:
        _log("  ⚠️  Fichier IPC introuvable")
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        _log(f"  ⚠️  IPC : {e}")
        return pd.DataFrame()

    # Filtrer les lignes HDX (où 'Date of analysis' commence par '#')
    if "Date of analysis" in df.columns:
        df = df[~df["Date of analysis"].astype(str).str.startswith("#")]
    elif "Country" in df.columns:
        df = df[~df["Country"].astype(str).str.startswith("#")]

    # Détecter les colonnes utiles
    region_col = next((c for c in df.columns
                      if "level 1" in c.lower() or c.lower() == "level 1"), None)
    area_col = next((c for c in df.columns
                    if c.lower() == "area"), None)
    phase_col = next((c for c in df.columns
                     if c.lower() == "phase"), None)
    num_col = next((c for c in df.columns
                   if c.lower() == "number"), None)
    pct_col = next((c for c in df.columns
                   if c.lower() == "percentage"), None)
    date_col = next((c for c in df.columns
                    if "date of analysis" in c.lower()), None)

    if not all([region_col, phase_col, num_col]):
        _log(f"  ⚠️  IPC : colonnes manquantes "
             f"(region={region_col}, phase={phase_col}, number={num_col})")
        return pd.DataFrame()

    df = df.rename(columns={
        region_col: "region",
        phase_col: "phase",
        num_col: "population",
        date_col or "date": "date_analyse",
    })
    if area_col:
        df = df.rename(columns={area_col: "zone"})

    if "zone" not in df.columns:
        df["zone"] = df["region"]

    df["region"] = df["region"].astype(str).str.strip().str.title()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["phase"] = df["phase"].astype(str).str.strip().str.lower()

    if pct_col:
        df = df.rename(columns={pct_col: "pct"})
        df["pct"] = pd.to_numeric(df["pct"], errors="coerce")

    _log(f"  ✓ IPC : {len(df)} enregistrements, "
         f"{df['region'].nunique()} zones, "
         f"phases : {sorted(df['phase'].unique())}")

    return df


# ════════════════════════════════════════════════════════════════
# 5. Dérivation indicateurs + alertes normatives
# ════════════════════════════════════════════════════════════════

def _derive_indicators(gdf):
    pop = gdf["population"].replace(0, np.nan)
    area = gdf["area_km2"].replace(0, np.nan)

    gdf["health_pop_per_fac"]     = pop / gdf["nb_sante"].replace(0, np.nan)
    gdf["health_fac_per_km2"]     = 100 * gdf["nb_sante"] / area
    gdf["health_fac_per_10000"]   = 10000 * gdf["nb_sante"] / pop
    gdf["water_pts_per1000"]      = 1000 * gdf["nb_eau"] / pop
    gdf["water_pts_per_km2"]      = 100 * gdf["nb_eau"] / area
    gdf["food_markets_per1000"]   = 1000 * gdf["nb_marches"] / pop
    gdf["food_markets_per_10000"] = 10000 * gdf["nb_marches"] / pop
    gdf["food_markets_dens"]      = 1000 * gdf["nb_marches"] / area
    #gdf["flood_fri_mean"]         = gdf["flood_fsi_max"]

    gdf["alerte_sante"] = gdf["health_fac_per_10000"]  < NORMES["sante"]["cible"]
    gdf["alerte_eau"]   = gdf["water_pts_per1000"]     < NORMES["eau"]["cible"]
    gdf["alerte_alim"]  = gdf["food_markets_per_10000"] < NORMES["alimentation"]["cible"]
    gdf["alerte_ino"]   = gdf["flood_fsi_max"]         > NORMES["inondation"]["cible"]
    gdf["alerte_ero"]   = gdf["erosion_epr_mean"]      < NORMES["erosion"]["cible"]
    
    # Intensité d'érosion = uniquement les valeurs négatives (érosion réelle)
# Convention source : EPR < 0 = recul du trait de côte ; EPR > 0 = accrétion
    if "erosion_epr_mean" in gdf.columns:
        epr = pd.to_numeric(gdf["erosion_epr_mean"], errors="coerce")
        gdf["erosion_intensite"] = (-epr).clip(lower=0)
    # NaN reste NaN (une donnée manquante n'est pas une absence d'érosion)
    return gdf


# ════════════════════════════════════════════════════════════════
# 6. Prix
# ════════════════════════════════════════════════════════════════

def _load_prices() -> pd.DataFrame:
    p = _find("prix_wfp")
    if not p:
        return pd.DataFrame(columns=["date", "marche", "denree", "prix_fcfa_kg"])
    try:
        df = pd.read_csv(p, low_memory=False)
        if "date" in df.columns:
            df = df[~df["date"].astype(str).str.startswith("#")]
        df = df.rename(columns={"commodity": "denree", "market": "marche", "price": "prix_fcfa_kg"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["prix_fcfa_kg"] = pd.to_numeric(df["prix_fcfa_kg"], errors="coerce")
        df = df.dropna(subset=["date", "prix_fcfa_kg", "denree", "marche"])
        _log(f"  → {len(df)} prix WFP")
        return df[["date", "marche", "denree", "prix_fcfa_kg"]].reset_index(drop=True)
    except Exception as e:
        _log(f"  ⚠️  prix : {e}")
        return pd.DataFrame(columns=["date", "marche", "denree", "prix_fcfa_kg"])


# ════════════════════════════════════════════════════════════════
# 7. API publique
# ════════════════════════════════════════════════════════════════

def _aggregate_projects_to_parents(levels: dict) -> dict:
    """Propage les colonnes projets (PCIAEPH/ZAAP) des communes vers préfectures/régions."""
    if "communes" not in levels:
        return levels

    communes = levels["communes"]
    project_cols = [c for c in communes.columns
                    if c.startswith(("pciaeph_", "zaap_"))]
    if not project_cols:
        return levels

    for parent, join_col in [("prefectures", "prefecture"), ("regions", "region")]:
        if parent not in levels or join_col not in communes.columns:
            continue
        if join_col not in levels[parent].columns:
            continue

        # Agréger : somme pour effectifs/coûts, moyenne pour taux/avancement
        agg_map = {c: "mean" if ("taux" in c or "avancement" in c) else "sum"
                   for c in project_cols}

        parent_agg = communes.groupby(join_col).agg(agg_map).reset_index()
        parent_agg = parent_agg.rename(columns={join_col: "_jk"})

        # Éviter les doublons de colonnes si déjà présentes
        existing = [c for c in project_cols if c in levels[parent].columns]
        if existing:
            levels[parent] = levels[parent].drop(columns=existing)

        levels[parent] = levels[parent].merge(
            parent_agg, left_on=join_col, right_on="_jk", how="left"
        ).drop(columns=["_jk"], errors="ignore")

        _log(f"  → Projets propagés vers {parent} "
             f"({levels[parent][project_cols].notna().any(axis=1).sum()}/{len(levels[parent])})")

    return levels
# ════════════════════════════════════════════════════════════════
# 4ter. Catalogue unifié de TOUS les projets COSO
# ════════════════════════════════════════════════════════════════

def _classify_sector(sector_raw: str, filename: str) -> str:
    """Normalise le secteur depuis le CSV ou le nom de fichier."""
    s = f"{sector_raw or ''} {filename}".lower()

    if any(k in s for k in ("eau", "hydraulique", "forage", "aep", "adduction")):
        return "Eau & hydraulique"
    if any(k in s for k in ("commerce", "marché", "marche", "market")):
        return "Commerce & marchés"
    if any(k in s for k in ("sant", "sanitaire",  "hôpital", "hopital", "clinique")):
        return "Santé"
    if any(k in s for k in ("educ", "école", "ecole", "school", "scolaire",
                             "collège", "college", "lycée", "lycee")):
        return "Éducation"
    if any(k in s for k in ("route", "piste", "road", "canal", "voirie")):
        return "Routes & pistes"
    if any(k in s for k in ("agri",  "élevage",  "culture", "récolte", "agricole", "irrig", "maraîch",
                             "maraich", "plantation")):
        return "Agriculture"
    if any(k in s for k in ("electr", "électr", "energ", "solaire", "solar",
                             "photovolt")):
        return "Énergie & électrification"
    if any(k in s for k in ("sport", "culture", "jeunesse", "loisir",
                             "stade", "terrain")):
        return "Sport & culture"
    if any(k in s for k in ("sécur", "secur", "police", "gendarmerie")):
        return "Sécurité"
    if any(k in s for k in ("santé-animale", "vétérin", "veterin", "élevage",
                             "elevage", "bétail", "betail")):
        return "Élevage & santé animale"
    return "Autre"


def _load_parity_education() -> pd.DataFrame:
    """Charge les indicateurs de parité Fille/Garçon par région."""
    candidates = []
    for d in (DATA_DIR, DATA_DIR_FALLBACK):
        if not Path(d).exists():
            continue
        for pat in ("edu_parite__*.csv", "*scolarisation*sexe*.csv",
                    "*eleves*sexe*.csv", "*parite*.csv"):
            candidates.extend(_data_matches(d, pat))

    if not candidates:
        _log("  ⚠️  Parité éducation : aucun fichier trouvé")
        return pd.DataFrame()

    # Prendre le premier fichier
    p = candidates[0]
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        _log(f"  ⚠️  {p.name} : {e}")
        return pd.DataFrame()

    # Détecter colonnes
    region_col = next((c for c in df.columns
                      if "region" in c.lower() or "région" in c.lower()), None)
    sexe_col = next((c for c in df.columns
                    if "sexe" in c.lower() or "genre" in c.lower()), None)
    value_col = next((c for c in df.columns
                     if c.lower() in ("value", "valeur", "taux")), None)
    date_col = next((c for c in df.columns
                    if c.lower() in ("date", "year", "annee")), None)

    if not all([region_col, sexe_col, value_col]):
        _log(f"  ⚠️  Parité : colonnes manquantes dans {p.name}")
        return pd.DataFrame()

    df = df.rename(columns={region_col: "region", sexe_col: "sexe",
                             value_col: "taux"})
    df["region"] = df["region"].astype(str).str.strip().str.title()
    df = df[df["region"].isin(REGIONS)]
    df["taux"] = pd.to_numeric(df["taux"], errors="coerce")

    # Prendre la valeur la plus récente par (region, sexe)
    if date_col:
        df = df.rename(columns={date_col: "annee"})
        df = df.sort_values("annee").groupby(["region", "sexe"]).tail(1)

    # Pivoter : une ligne par région avec 2 colonnes F et G
    try:
        pivot = df.pivot_table(index="region", columns="sexe",
                                values="taux", aggfunc="mean").reset_index()
    except Exception:
        return pd.DataFrame()

    # Renommer colonnes (variantes Filles/Féminin/F, Garçons/Masculin/M)
    rename_map = {}
    for c in pivot.columns:
        if c == "region":
            continue
        cl = str(c).lower()
        if any(k in cl for k in ("fille", "féminin", "feminin", "f", "femme")):
            rename_map[c] = "taux_filles"
        elif any(k in cl for k in ("garçon", "garcon", "masculin", "m", "homme")):
            rename_map[c] = "taux_garcons"
    pivot = pivot.rename(columns=rename_map)

    if "taux_filles" in pivot.columns and "taux_garcons" in pivot.columns:
        pivot["ratio_fg"] = (pivot["taux_filles"]
                             / pivot["taux_garcons"].replace(0, np.nan) * 100).round(1)
        pivot["ecart_fg"] = (pivot["taux_filles"] - pivot["taux_garcons"]).round(1)
        _log(f"  ✓ Parité éducation : {len(pivot)} régions")
    else:
        _log(f"  ⚠️  Parité : colonnes F/G non identifiées ({list(pivot.columns)})")
        return pd.DataFrame()

    return pivot

def _load_teachers() -> pd.DataFrame:
    """Charge les effectifs d'enseignants par région."""
    candidates = []
    for d in (DATA_DIR, DATA_DIR_FALLBACK):
        if not Path(d).exists():
            continue
        for pat in ("edu_enseignant*.csv", "*enseignant*region*.csv",
                    "*professeur*.csv", "*personnel*enseignant*.csv"):
            candidates.extend(_data_matches(d, pat))
    
    if not candidates:
        _log("  ⚠️  Enseignants : aucun fichier trouvé")
        return pd.DataFrame()
    
    p = candidates[0]
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        _log(f"  ⚠️  {p.name} : {e}")
        return pd.DataFrame()
    
    region_col = next((c for c in df.columns
                      if "region" in c.lower() or "région" in c.lower()), None)
    ens_col = next((c for c in df.columns
                   if "enseignant" in c.lower() or "professeur" in c.lower()
                   or "effectif" in c.lower()), None)
    value_col = next((c for c in df.columns
                     if c.lower() in ("value", "valeur", "nombre", "count", "total")), None)
    
    if not region_col or not (ens_col or value_col):
        _log(f"  ⚠️  Enseignants : colonnes non identifiées")
        return pd.DataFrame()
    
    df = df.rename(columns={region_col: "region",
                             (ens_col or value_col): "nb_enseignants"})
    df["region"] = df["region"].astype(str).str.strip().str.title()
    df = df[df["region"].isin(REGIONS)]
    df["nb_enseignants"] = pd.to_numeric(df["nb_enseignants"], errors="coerce")
    
    agg = df.groupby("region")["nb_enseignants"].sum().reset_index()
    _log(f"  ✓ Enseignants : {len(agg)} régions")
    return agg

def load_data():
    """Retourne (levels_dict, prices_df, is_demo)."""
    _log("═" * 60)
    result = _load_levels_from_fri()
    if result is None:
        _log("❌ Aucun niveau → mode DÉMO")
        return {}, pd.DataFrame(), True

    levels, fri = result
    communes = _load_communes(fri)
    if communes is not None:
        levels["communes"] = communes

    _log("Chargement des points…")
    pts_simple, pts_attr = _load_points()
    for name in list(levels.keys()):
        if name.startswith("_"):
            continue
        levels[name] = _aggregate_points(levels[name], pts_simple, pts_attr)

    # Enrichir avec PCIAEPH (uniquement sur communes)
        # Enrichir avec PCIAEPH (uniquement sur communes)
    if "communes" in levels:
        pciaeph = _load_pciaeph(levels["communes"])
        if not pciaeph.empty:
            levels["communes"] = levels["communes"].merge(pciaeph, on="id", how="left")
            _log(f"  → PCIAEPH : {len(pciaeph)} communes enrichies")

        # Projets détaillés (statuts, coûts, bénéficiaires)
        pciaeph_det = _load_pciaeph_detailed(levels["communes"])
        if not pciaeph_det.empty:
            levels["communes"] = levels["communes"].merge(
                pciaeph_det, on="id", how="left")
            _log(f"  → PCIAEPH détaillé fusionné")

        # Zones agricoles
        zaaps = _load_zaaps(levels["communes"])
        
    # ── Catalogue unifié de TOUS les projets
    if "communes" in levels:
        all_projects = _load_all_projects(levels["communes"])
        if not all_projects.empty:
            levels["_projects_catalog"] = all_projects
            _log(f"  → Catalogue projets : {len(all_projects)} entrées "
                 f"({all_projects['secteur'].nunique()} secteurs)") 
                          
    # ── Propager les projets vers préfectures et régions
    levels = _aggregate_projects_to_parents(levels)
    if not zaaps.empty:
        levels["communes"] = levels["communes"].merge(zaaps, on="id", how="left")
        _log(f"  → ZAAPs fusionnés")
    
    # ── Nettoyage final : fusionner les colonnes dupliquées _x / _y        
    for name in list(levels.keys()):
        if name.startswith("_"):
            continue
        g = levels[name]
        for base in ("pciaeph_avancement", "pciaeph_cout",
                     "pciaeph_profondeur", "pciaeph_debit",
                     "zaap_nb", "zaap_surface_ha"):
            cols = [c for c in g.columns
                    if c == base or c.startswith(f"{base}_")]
            if len(cols) > 1:
                # Fusionner d'abord
                merged_values = g[cols].bfill(axis=1).iloc[:, 0]
                # Puis supprimer TOUTES les colonnes (y compris _x)
                g = g.drop(columns=cols)
                # Enfin recréer la colonne unifiée
                g[base] = merged_values
        levels[name] = g
        
            # ── Alimentation : bilan vivrier + IPC
    bilan = _load_bilan_vivrier()
    ipc = _load_ipc()

    if not bilan.empty:
        levels["_bilan_vivrier"] = bilan
        _log(f"  → Bilan vivrier : {len(bilan)} enregistrements")

    if not ipc.empty:
        levels["_ipc"] = ipc
        _log(f"  → IPC : {len(ipc)} enregistrements")


          # ── Éducation : indicateurs régionaux + écoles par type + enseignants
    edu_ind = _load_education_indicators()
    teachers = _load_teachers_by_region()

    if not edu_ind.empty:
        # Fusionner avec le niveau régions
        if "regions" in levels and "region" in levels["regions"].columns:
            levels["regions"] = levels["regions"].merge(
                edu_ind.rename(columns={"region": "region"}),
                on="region", how="left"
            )
            _log(f"  → Indicateurs éducation : fusionnés sur {len(levels['regions'])} régions")

    if not teachers.empty and "regions" in levels:
        levels["regions"] = levels["regions"].merge(teachers, on="region", how="left")
        _log(f"  → Enseignants : fusionnés sur régions")

    if "communes" in levels:
        schools_by_type = _load_schools_by_type(levels["communes"])
        if not schools_by_type.empty:
            # Ajouter une colonne _key aux communes pour la jointure
            levels["communes"]["_key"] = levels["communes"]["commune"].map(_norm)
            # Adapter le nom de la clé de jointure ('commune_key' → '_key')
            if "commune_key" in schools_by_type.columns:
                schools_by_type = schools_by_type.rename(columns={"commune_key": "_key"})
            levels["communes"] = levels["communes"].merge(
                schools_by_type, on="_key", how="left"
            )
            levels["communes"] = levels["communes"].drop(columns=["_key"], errors="ignore")
            # Remplir les NaN par 0
            for col in ["nb_maternelle", "nb_primaire", "nb_college", "nb_lycee"]:
                if col in levels["communes"].columns:
                    levels["communes"][col] = levels["communes"][col].fillna(0).astype(int)
            _log(f"  → Écoles par type : fusionnées sur communes")


    for name in list(levels.keys()):
        if name.startswith("_"):
            continue
        levels[name] = _derive_indicators(levels[name])

    prices = _load_prices()
    _log(f"✅ {len(levels)} niveaux : "
         + ", ".join(f"{k}({len(v)})" for k, v in levels.items()))
    _log("═" * 60)
    
        # ── Parité Fille/Garçon
    parity = _load_parity_education()
    if not parity.empty:
        levels["_parity_education"] = parity
        _log(f"  → Parité éducation : {len(parity)} régions")
        
        # ── Enseignants
    teachers = _load_teachers()
    if not teachers.empty and "regions" in levels:
        levels["regions"] = levels["regions"].merge(teachers, on="region", how="left")
        _log(f"  → Enseignants : fusionnés sur {len(levels['regions'])} régions")
        
    return levels, prices, False
    


def diagnose():
    import json
    report = {"fichiers": {}, "niveaux": {}, "alertes_qualite": [], "erreurs": []}
    for key in FILES:
        p = _find(key)
        report["fichiers"][key] = os.path.basename(p) if p else "❌ introuvable"
    try:
        levels, prices, is_demo = load_data()
        report["is_demo"] = is_demo
        for name, gdf in levels.items():
            if name.startswith("_"):
                continue  # skip _projects_catalog etc.
            report["niveaux"][name] = {
                "n": len(gdf),
                "colonnes": [c for c in gdf.columns if c != "geometry"],
            }
        report["prix"] = len(prices)
    except Exception as e:
        report["erreurs"].append(f"load_data : {e}")
        import traceback
        traceback.print_exc()

    try:
        report["alertes_qualite"] = _check_pciaeph_coherence()
    except Exception as e:
        report["erreurs"].append(f"audit PCIAEPH : {e}")

    return report


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2, ensure_ascii=False, default=str))