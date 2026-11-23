# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  VIGIE·TOGO 360 — Observatoire de la Vulnérabilité                   ║
║                    et de la Résilience Territoriale                  ║
║  Version 4.0 — Navigation cliquable, KPI interactifs, guide novice   ║
╚══════════════════════════════════════════════════════════════════════╝

📘 POUR LES DÉBUTANTS — COMMENT LIRE CE FICHIER
────────────────────────────────────────────────
Ce fichier est le "chef d'orchestre" de l'application. Il est organisé
en 9 blocs, toujours dans le même ordre :

  0. Configuration     → titre de l'onglet, couleurs, noms des piliers
  1. Icônes SVG        → petits dessins vectoriels réutilisables
  2. Style CSS         → toute l'apparence (couleurs, animations…)
  3. Données           → chargement + calcul de l'indice ICV (mis en cache)
  4. Composants UI     → briques réutilisables : bandeau, carte KPI, carte…
  5. Barre latérale    → choix du niveau, filtres, pondération
  6. Pages             → UNE fonction par page de l'application
  7. Navigation        → le menu, regroupé par thème
  8. Pied de page

📘 LES NOUVEAUTÉS DE CETTE VERSION
────────────────────────────────────
  ✔ Chaque KPI / tuile est CLIQUABLE et ouvre la page liée
    (et applique au passage un filtre, ex. "seulement les critiques").
  ✔ Cliquer sur un territoire (carte, barre, ligne de tableau) propose
    d'ouvrir sa fiche territoriale.
  ✔ Accueil repensé : tuiles de navigation, alerte, top 5, répartition.
  ✔ Menu regroupé en 3 familles ; logo placé en haut de la barre latérale.
  ✔ Corrections de bugs (voir la liste à la fin de ce fichier).

📘 PRÉREQUIS
─────────────
  pip install -U streamlit streamlit-folium folium geopandas plotly scipy
  → Streamlit ≥ 1.39 est nécessaire (conteneurs avec clé, pour les KPI
    cliquables). Lancement :  streamlit run app.py
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd  # noqa: F401  (utilisé par les modules de données)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

# Modules du projet (dans le même dossier que app.py)
from data_loader import (load_data, PILLARS, INDICATORS, NORMES,  # noqa: F401
                         get_sources_registry, _check_pciaeph_coherence)
from icv import compute_pillars, compute_icv, double_peine, prioritization_quadrant

# On masque quelques avertissements techniques sans gravité pour garder
# une console propre.
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*will be removed.*")
warnings.filterwarnings("ignore", message=".*CartoDB.*")

# ════════════════════════════════════════════════════════════════
# 0. CONFIGURATION & IDENTITÉ VISUELLE
# ════════════════════════════════════════════════════════════════
APP_NAME = "VIGIE·TOGO 360"
APP_TAGLINE = "Observatoire de la Vulnérabilité et de la Résilience Territoriale"

# ⚠️ set_page_config doit TOUJOURS être la 1ère commande Streamlit.
st.set_page_config(
    page_title=f"{APP_NAME} — Atlas de la vulnérabilité",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Les piliers ICV actifs : nom affiché + couleur associée.
PILLAR_LABEL = {"sante": "Santé", "eau": "Eau", "alimentation": "Alimentation",
                "inondation": "Inondation", "erosion": "Érosion côtière"}
PILLAR_COLOR = {"sante": "#C8102E", "eau": "#0e7490", "alimentation": "#d97706",
                "inondation": "#0284c7", "erosion": "#92400e"}
PILLAR_DESC = {
    "sante": "Accès aux formations sanitaires",
    "eau": "Accès à l'eau potable",
    "alimentation": "Accès aux marchés",
    "inondation": "Risque d'inondation",
    "erosion": "Recul du littoral",
}

# Couleurs du drapeau togolais (vert · jaune · rouge) + neutres.
C_GREEN_DK = "#00563f"
C_GREEN = "#006a4e"
C_GREEN_LT = "#e6f2ec"
C_YELLOW = "#FFCE00"
C_RED = "#C8102E"
C_RED_LT = "#fde8ea"
C_INK = "#14211c"
C_INK_SOFT = "#4b5a54"
C_LINE = "#e2e8e5"

# Couleur de chaque classe de vulnérabilité (pour les graphiques).
CLASS_COLOR = {"Critique": C_RED, "Élevée": "#f59e0b",
               "Modérée": "#eab308", "Faible": "#10b981"}

# Les mailles territoriales : libellé affiché → clé dans `levels`.
# (BUG CORRIGÉ : avant, "Régions".lower() donnait "régions" avec accent,
#  ce qui ne correspondait à aucune clé du dictionnaire de données.)
NIVEAUX = {"Régions": "regions", "Préfectures": "prefectures",
           "Communes": "communes", "Cantons": "cantons"}

# Version des données : changez ce nombre pour forcer un rechargement.
DATA_VERSION = 6
MATRIX_SERVICES_THRESHOLD = 60
MATRIX_FLOOD_THRESHOLD = 55

# ════════════════════════════════════════════════════════════════
# 1. ICÔNES SVG
# ════════════════════════════════════════════════════════════════
# 📘 Un SVG est un dessin "mathématique" : net à toutes les tailles et
#    colorable en CSS (currentColor = couleur du texte autour).
def _svg(paths: str, extra: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'width="1em" height="1em" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            f'{extra}>{paths}</svg>')


ICON = {
    "logo": _svg('<path d="M12 2l7 3v6c0 5-3.5 8.5-7 11-3.5-2.5-7-6-7-11V5l7-3z"/><path d="M9 12l2 2 4-4"/>'),
    "home": _svg('<path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/>'),
    "sante": _svg('<path d="M12 21s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.5-7 10-7 10z"/>'),
    "eau": _svg('<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/>'),
    "alimentation": _svg('<path d="M12 22V10"/><path d="M12 10c0-3 2-5 5-5 0 3-2 5-5 5z"/>'),
    "inondation": _svg('<path d="M7 10l5-5 5 5"/><path d="M12 5v8"/>'),
    "erosion": _svg('<path d="M3 20h18"/><path d="M5 20c2-4 5-6 9-6s5 2 7 4"/>'),
    "dashboard": _svg('<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>'),
    "analyse": _svg('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>'),
    "scenario": _svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'),
    "projets": _svg('<rect x="4" y="10" width="16" height="10" rx="1"/><path d="M8 10V6h8v4"/>'),
    "fiche": _svg('<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>'),
    "prix": _svg('<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>'),
    "qualite": _svg('<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'),
    "sources": _svg('<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5z"/>'),
    "methode": _svg('<path d="M9 3h6"/><path d="M10 3v6L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9V3"/>'),
    "map": _svg('<path d="M9 4L3 6v14l6-2 6 2 6-2V4l-6 2-6-2z"/><path d="M9 4v14M15 6v14"/>'),
    "filter": _svg('<path d="M3 5h18l-7 8v6l-4 2v-8L3 5z"/>'),
    "scale": _svg('<path d="M12 3v18"/><path d="M5 8l7-5 7 5"/><path d="M3 14l3-6 3 6a3 3 0 0 1-6 0zM15 14l3-6 3 6a3 3 0 0 1-6 0z"/>'),
    "pop": _svg('<circle cx="9" cy="8" r="3.5"/><circle cx="17" cy="9" r="2.5"/><path d="M2 20c0-3.5 3-6 7-6s7 2.5 7 6M16 14c3 0 6 2 6 5"/>'),
    "alert": _svg('<path d="M12 3l10 17H2L12 3z"/><path d="M12 10v4M12 17.5v.01"/>'),
    "check": _svg('<path d="M4 12l5 5L20 6"/>'),
    "info": _svg('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/>'),
    "print": _svg('<path d="M6 9V3h12v6"/><rect x="6" y="14" width="12" height="8"/><path d="M6 18H4a1 1 0 0 1-1-1v-6a1 1 0 0 1 1-1h16a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1h-2"/>'),
}


def ic(name: str) -> str:
    """Retourne le HTML d'une icône (ex. ic("map"))."""
    return f'<span class="ic">{ICON[name]}</span>'


# Logo affiché tout en haut de la barre latérale (st.logo veut un fichier).
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 270 52" width="270" height="52">'
    '<rect x="0" y="4" width="44" height="44" rx="11" fill="#006a4e"/>'
    '<rect x="0" y="4" width="5" height="44" rx="2.5" fill="#C8102E"/>'
    '<g transform="translate(8 12) scale(1.2)" fill="none" stroke="#fff" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2l7 3v6c0 5-3.5 8.5-7 11-3.5-2.5-7-6-7-11V5l7-3z"/><path d="M9 12l2 2 4-4"/></g>'
    '<text x="56" y="28" font-family="Segoe UI,Arial,sans-serif" font-size="21" font-weight="800" '
    'fill="#14211c">VIGIE<tspan fill="#FFCE00">·</tspan>TOGO 360</text>'
    '<text x="56" y="43" font-family="Segoe UI,Arial,sans-serif" font-size="9.5" fill="#4b5a54">'
    'Atlas de la vulnérabilité territoriale</text></svg>'
)


def _ensure_logo() -> str | None:
    """Écrit le logo dans ./assets/logo.svg s'il n'existe pas encore."""
    try:
        p = Path(__file__).parent / "assets" / "logo.svg"
        p.parent.mkdir(exist_ok=True)
        p.write_text(LOGO_SVG, encoding="utf-8")
        return str(p)
    except Exception:
        return None


_logo_path = _ensure_logo()
if _logo_path:
    try:
        st.logo(_logo_path, size="large")
    except Exception:
        pass  # anciennes versions de Streamlit : on ignore le logo

# ════════════════════════════════════════════════════════════════
# 2. STYLE CSS
# ════════════════════════════════════════════════════════════════
# 📘 Le CSS décrit l'apparence. Les "variables" (--green, --ink…) évitent
#    de répéter les couleurs : changez-en une, tout l'écran suit.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');
:root{--green:#006a4e; --green-dk:#00563f; --green-lt:#e6f2ec; --yellow:#FFCE00;
  --red:#C8102E; --red-lt:#fde8ea; --ink:#14211c; --ink-soft:#4b5a54; --line:#e2e8e5;}
.stApp{font-family:'Inter','Segoe UI',sans-serif;}
h1,h2,h3,h4{font-family:'Sora','Inter',sans-serif;}
.block-container{padding-top:1rem; padding-bottom:3rem; max-width:1600px;}
.ic{display:inline-flex; vertical-align:-.15em; line-height:1;}
.ic svg{width:1em; height:1em;}

/* ---------- Animations ---------- */
@keyframes fadeUp{from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;}}
@keyframes shimmer{0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;}}
@keyframes pulseSoft{0%,100%{box-shadow:0 0 0 0 rgba(200,16,46,.35);} 50%{box-shadow:0 0 0 9px rgba(200,16,46,0);}}
@keyframes floatY{0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);}}
@keyframes drift{0%{transform:translate(0,0) scale(1);} 50%{transform:translate(-30px,18px) scale(1.12);} 100%{transform:translate(0,0) scale(1);}}
@keyframes sweep{from{background-position:-200% 0;} to{background-position:200% 0;}}
.fade-up{animation:fadeUp .6s cubic-bezier(.2,.7,.2,1) both;}
/* Apparition en cascade : chaque colonne démarre un peu après la précédente */
[data-testid="stColumn"]:nth-child(2) .fade-up{animation-delay:.07s;}
[data-testid="stColumn"]:nth-child(3) .fade-up{animation-delay:.14s;}
[data-testid="stColumn"]:nth-child(4) .fade-up{animation-delay:.21s;}
[data-testid="stColumn"]:nth-child(5) .fade-up{animation-delay:.28s;}
@media (prefers-reduced-motion:reduce){*{animation:none !important; transition:none !important;}}

/* ---------- Bandeau héros ---------- */
.hero{position:relative; overflow:hidden; border-radius:20px; margin-bottom:1.2rem;
  background:linear-gradient(120deg,#003d2e 0%,#006a4e 48%,#00805e 100%);
  background-size:200% 200%; animation:shimmer 10s ease infinite;
  color:#fff; padding:1.9rem 2rem 1.7rem 6.4rem;
  box-shadow:0 14px 36px rgba(0,90,63,.30); border-left:6px solid var(--yellow);}
.hero::before{content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--red);}
.hero::after{content:""; position:absolute; right:-70px; top:-90px; width:280px; height:280px;
  border-radius:50%; background:radial-gradient(circle,rgba(255,206,0,.30),transparent 68%);
  animation:drift 12s ease-in-out infinite;}
.hero .stripe{position:absolute; left:0; right:0; bottom:0; height:5px;
  background:linear-gradient(90deg,var(--green) 0 33%,var(--yellow) 33% 66%,var(--red) 66%);}
.hero .logo-mark{position:absolute; left:1.6rem; top:50%; transform:translateY(-50%);
  width:3.5rem; height:3.5rem; border-radius:14px; background:#fff;
  display:flex; align-items:center; justify-content:center; color:var(--green);
  box-shadow:0 6px 18px rgba(0,0,0,.28); animation:floatY 4s ease-in-out infinite;}
.hero .logo-mark svg{width:2.1rem; height:2.1rem;}
.hero h1{margin:0; font-family:'Sora',sans-serif; font-size:1.75rem; font-weight:800; color:#fff;}
.hero h1 .dot{color:var(--yellow);}
.hero p{margin:.4rem 0 0; color:rgba(255,255,255,.93); font-size:.95rem; line-height:1.55;}
.hero .chip-row{margin-top:.85rem; display:flex; gap:.5rem; flex-wrap:wrap; position:relative; z-index:1;}
.hero .chip{background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.28);
  color:#fff; padding:.28rem .75rem; border-radius:999px; font-size:.75rem; font-weight:600;
  backdrop-filter:blur(4px);}

/* ---------- Cartes KPI ---------- */
.kpi{position:relative; background:#fff; border:1px solid var(--line); border-top:4px solid var(--green);
  padding:1rem 1.15rem; border-radius:16px; height:100%; box-sizing:border-box;
  box-shadow:0 2px 8px rgba(20,33,28,.05);
  transition:transform .22s cubic-bezier(.2,.7,.2,1), box-shadow .22s, border-color .22s;}
.kpi .label{font-size:.7rem; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.08em;
  font-weight:700; display:flex; align-items:center; gap:.35rem;}
.kpi .value{font-family:'Sora',sans-serif; font-size:1.75rem; font-weight:800;
  margin:.3rem 0 .1rem; line-height:1.05; color:var(--ink);}
.kpi .sub{font-size:.76rem; color:var(--ink-soft); line-height:1.4;}
.kpi .bar{height:5px; background:var(--line); border-radius:99px; margin-top:.65rem; overflow:hidden;}
.kpi .bar > i{display:block; height:100%; background:var(--green); border-radius:99px;}
.kpi .go{margin-top:.7rem; font-size:.74rem; font-weight:700; color:var(--green);
  opacity:.55; transition:opacity .2s, transform .2s;}
.kpi.danger{border-top-color:var(--red);}
.kpi.danger .value{color:var(--red);}
.kpi.danger .bar > i{background:var(--red);}
.kpi.danger .go{color:var(--red);}
.kpi.danger.pulse{animation:pulseSoft 2.2s infinite;}
.kpi.warn{border-top-color:#d97706;}
.kpi.warn .value{color:#b45309;}
.kpi.warn .bar > i{background:#d97706;}
.kpi.warn .go{color:#b45309;}
.kpi.ok{border-top-color:#059669;}
.kpi.ok .value{color:#047857;}

/* ---------- Tuiles de navigation & bannière ---------- */
.tile{display:flex; align-items:center; gap:.9rem; background:#fff; border:1px solid var(--line);
  border-radius:16px; padding:1rem 1.1rem; height:100%; box-sizing:border-box;
  box-shadow:0 2px 8px rgba(20,33,28,.04);
  transition:transform .22s cubic-bezier(.2,.7,.2,1), box-shadow .22s, border-color .22s;}
.tile .tile-ic{width:2.7rem; height:2.7rem; border-radius:12px; background:var(--green-lt);
  color:var(--green); display:flex; align-items:center; justify-content:center;
  font-size:1.3rem; flex:none; transition:background .2s, color .2s;}
.tile b{display:block; font-size:.95rem; color:var(--ink);}
.tile .d{display:block; font-size:.76rem; color:var(--ink-soft); line-height:1.35; margin-top:.1rem;}
.tile .arrow{margin-left:auto; color:var(--green); font-weight:800; transition:transform .2s;}
.banner{display:flex; align-items:center; gap:.9rem; padding:1rem 1.3rem; border-radius:16px;
  background:linear-gradient(100deg,#fde8ea,#fff5f5); border:1px solid #f5b5bd;
  border-left:6px solid var(--red); color:#7f1d1d; font-size:.92rem;
  transition:transform .22s, box-shadow .22s;}
.banner .ic{font-size:1.6rem; color:var(--red);}
.banner .arrow{margin-left:auto; font-weight:800; color:var(--red); white-space:nowrap;}

/* 📘 ASTUCE KPI CLIQUABLE : chaque carte est dans un conteneur Streamlit dont
   la clé commence par "kpi_". Un vrai bouton Streamlit y est placé, rendu
   invisible (opacity:0) et étiré sur TOUTE la carte → toute la carte est un bouton. */
[class*="st-key-kpi_"]{position:relative; height:100%; gap:0 !important;}
[class*="st-key-kpi_"] [data-testid="stElementContainer"]:has(button),
[class*="st-key-kpi_"] .element-container:has(button){
  position:absolute !important; inset:0; z-index:5; margin:0 !important; width:100% !important;}
[class*="st-key-kpi_"] [data-testid="stButton"], [class*="st-key-kpi_"] .stButton{height:100%;}
[class*="st-key-kpi_"] button{position:absolute !important; inset:0; width:100% !important;
  height:100% !important; opacity:0; cursor:pointer; padding:0 !important; min-height:0 !important;}
[class*="st-key-kpi_"]:hover .kpi, [class*="st-key-kpi_"]:hover .tile{
  transform:translateY(-5px); box-shadow:0 16px 30px rgba(0,106,78,.18); border-color:var(--green);}
[class*="st-key-kpi_"]:hover .kpi .go{opacity:1; transform:translateX(4px);}
[class*="st-key-kpi_"]:hover .tile .arrow{transform:translateX(5px);}
[class*="st-key-kpi_"]:hover .tile .tile-ic{background:var(--green); color:#fff;}
[class*="st-key-kpi_"]:hover .banner{transform:translateY(-3px); box-shadow:0 12px 26px rgba(200,16,46,.2);}
[class*="st-key-kpi_"]:active .kpi, [class*="st-key-kpi_"]:active .tile{transform:scale(.985);}

/* ---------- Titres de section & encadrés ---------- */
.section-title{font-family:'Sora',sans-serif; font-size:1.08rem; font-weight:700;
  color:var(--ink); margin:1.6rem 0 .8rem; padding-left:.75rem;
  border-left:4px solid var(--green); line-height:1.2; display:flex; align-items:center; gap:.5rem;}
.section-title .ic{color:var(--green);}
.box{padding:.9rem 1.1rem; border-radius:12px; margin:.8rem 0; font-size:.88rem;
  line-height:1.65; border-left:4px solid; display:flex; gap:.6rem; align-items:flex-start;}
.box.info{background:#eff6ff; border-color:#0284c7; color:#0c4a6e;}
.box.tip{background:#fffbeb; border-color:#d97706; color:#78350f;}
.box.alert{background:var(--red-lt); border-color:var(--red); color:#7f1d1d;}
.box.ok{background:#ecfdf5; border-color:#059669; color:#064e3b;}
.badge{display:inline-flex; align-items:center; gap:.3rem; padding:.22rem .7rem;
  border-radius:999px; font-size:.72rem; font-weight:700;}
.badge .dot{width:8px; height:8px; border-radius:50%; background:currentColor;}
.badge-critique{background:var(--red-lt); color:var(--red);}
.badge-eleve{background:#fef3c7; color:#92400e;}
.badge-modere{background:#fef9c3; color:#713f12;}
.badge-faible{background:#d1fae5; color:#065f46;}
.crumb{font-size:.78rem; color:var(--ink-soft); margin:.2rem 0 .5rem;}
.crumb b{color:var(--green-dk);}

/* ---------- Boutons, onglets ---------- */
.stButton button{border-radius:10px; font-weight:600; transition:all .15s;}
.stButton button:hover{box-shadow:0 6px 16px rgba(0,106,78,.18); transform:translateY(-1px);}
.stTabs [data-baseweb="tab-list"]{gap:.4rem; border-bottom:2px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:.55rem 1.1rem; border-radius:10px 10px 0 0;
  font-weight:600; font-size:.9rem; color:var(--ink-soft);}
.stTabs [aria-selected="true"]{color:var(--green-dk) !important; background:var(--green-lt);}

/* ---------- Barre latérale ---------- */
[data-testid="stSidebar"]{background:#fff; border-right:1px solid var(--line);}
[data-testid="stSidebarNavLink"]{border-radius:10px; margin:1px 0;}
[data-testid="stSidebarNavLink"]:hover{background:var(--green-lt);}
[data-testid="stSidebarNavLink"][aria-current="page"]{
  background:var(--green-lt); border-left:4px solid var(--green); font-weight:700;}
[data-testid="stSidebarNavSeparator"], [data-testid="stNavSectionHeader"]{
  color:var(--green-dk) !important; font-weight:800 !important; letter-spacing:.04em;}
.side-h{display:flex; align-items:center; gap:.5rem; font-family:'Sora',sans-serif;
  font-size:.92rem; font-weight:700; color:#00563f; margin:1.1rem 0 .5rem;
  padding-bottom:.3rem; border-bottom:1px dashed #e2e8e5;}
.side-h .ic{color:#006a4e; font-size:1rem;}

/* ---------- Fiche citoyenne & pied de page ---------- */
.fiche-citoyen{background:#fff; border:2px solid var(--green); border-radius:16px;
  padding:1.6rem 1.8rem; box-shadow:0 4px 18px rgba(20,33,28,.08); max-width:820px; margin:0 auto;}
.fiche-big{font-family:'Sora'; font-size:2.6rem; font-weight:800; color:var(--green);
  text-align:center; line-height:1;}
.fiche-big small{display:block; font-size:.8rem; font-weight:500; color:var(--ink-soft); margin-top:.3rem;}
.fiche-reco{background:var(--green-lt); border-left:4px solid var(--green);
  padding:.9rem 1.1rem; border-radius:10px; margin-top:1rem; font-size:.9rem;}
.app-footer{margin-top:2.2rem; padding-top:1rem; border-top:1px solid var(--line);
  font-size:.76rem; color:var(--ink-soft); text-align:center; line-height:1.7;}

/* ---------- Impression : on cache menus et boutons ---------- */
@media print{
  [data-testid="stSidebar"], [data-testid="stHeader"], .stButton, .stTabs [role="tablist"]{display:none !important;}
  .block-container{max-width:100% !important; padding:0 !important;}
}


/* ── Header Streamlit : on le rend discret et non chevauchant ── */
header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.95) !important;
    height: 2.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.block-container {
    padding-top: 2.75rem !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 3. CHARGEMENT & CALCUL DES DONNÉES (mis en cache)
# ════════════════════════════════════════════════════════════════
# 📘 Le "cache" mémorise le résultat d'une fonction coûteuse. Sans lui, les
#    données seraient rechargées à CHAQUE clic. Streamlit relance tout le script
#    à chaque interaction : le cache est donc indispensable.
#
# ⚠️ BUG CORRIGÉ : avant, le paramètre s'appelait `_v`. Or Streamlit IGNORE les
#    paramètres commençant par "_" pour décider d'invalider le cache : changer
#    `_v` ne rechargeait donc jamais rien. On l'appelle désormais `version`.
@st.cache_data(show_spinner="Chargement des données…")
def get_data(version: int = DATA_VERSION):
    levels_, prices_, is_demo_ = load_data()
    for k, df in list(levels_.items()):
        if k.startswith("_"):            # tables annexes (ex. catalogue projets)
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "population" not in df.columns:
            df = df.copy()
            df["population"] = float("nan")
        df = compute_pillars(df)                                   # 4 scores 0-100
        df = compute_icv(df, {p: 0.25 for p in PILLARS})           # ICV de base
        df = double_peine(df)                                      # drapeau "double peine"
        levels_[k] = df
    return levels_, prices_, is_demo_


levels, prices, is_demo = get_data(DATA_VERSION)


@st.cache_data(show_spinner=False, max_entries=40)
def compute_icv_cached(niveau: str, weights: tuple, version: int):
    """Recalcule l'ICV pour un niveau et des poids donnés, avec mémorisation.

    📘 Si vous replacez les curseurs sur une valeur déjà vue, le résultat
       revient instantanément, sans recalcul."""
    return compute_icv(levels[niveau].copy(), dict(weights))


# ════════════════════════════════════════════════════════════════
# 4. COMPOSANTS D'INTERFACE (briques réutilisables)
# ════════════════════════════════════════════════════════════════
# Registre des pages : rempli au §7, utilisé ici pour les liens cliquables.
NAV_PAGES: dict[str, st.Page] = {}
PAGE_TITLES: dict[str, str] = {}

# Compteur qui donne un identifiant unique à chaque carte cliquable.
# Il est remis à zéro à chaque exécution du script (donc stable d'un clic à l'autre).
_UID = [0]

# Page actuellement affichée (clé du §7) : sert à l'aide contextuelle de hero().
CURRENT: dict = {"key": None}


def clickable(html: str, target: str | None = None, state: dict | None = None,
              tooltip: str | None = None) -> None:
    """Affiche `html` ; s'il y a une `target`, toute la zone devient un bouton.

    📘 PARAMÈTRES
       html    : le code HTML de la carte à afficher
       target  : clé de la page à ouvrir au clic (ex. "dashboard") — voir §7
       state   : valeurs à mémoriser AVANT d'ouvrir la page (filtres, sélection…)
       tooltip : bulle d'aide au survol
    """
    if not target or target not in NAV_PAGES:
        st.markdown(html, unsafe_allow_html=True)
        return
    _UID[0] += 1
    uid = f"{re.sub(r'[^a-z0-9]', '', target)}_{_UID[0]}"
    with st.container(key=f"kpi_{uid}"):                 # classe CSS st-key-kpi_…
        st.markdown(html, unsafe_allow_html=True)
        if st.button("Ouvrir", key=f"kpibtn_{uid}",
                     help=tooltip or f"Ouvrir : {PAGE_TITLES.get(target, target)}"):
            for k, v in (state or {}).items():
                st.session_state[k] = v                   # ex. focus = "critique"
            st.switch_page(NAV_PAGES[target])             # redirection


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    """Grand bandeau vert en haut de chaque page (logo + titre + puces)."""
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(
        f'<div class="hero fade-up"><div class="logo-mark">{ICON["logo"]}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p>'
        f'<div class="chip-row">{chips_html}</div><div class="stripe"></div></div>',
        unsafe_allow_html=True)
    _contextual_help()          # bandeau pliable « Comment lire cette page ? »


def kpi_card(label, value, sub="", variant="", pct=None, icon=None, pulse=False,
             target=None, state=None, hint=None) -> None:
    """Carte indicateur (KPI). Devient CLIQUABLE si `target` est renseigné.

    📘 EXEMPLE
       kpi_card("Critiques", 12, "ICV > 75", "danger", icon="alert",
                target="dashboard", state={"focus": "critique"})
       → au clic : ouvre le Tableau de bord, filtré sur les territoires critiques.

    variant : "" (vert) · "danger" (rouge) · "warn" (orange) · "ok" (vert foncé)
    pct     : si fourni (0-100), affiche une barre de progression
    """
    bar = ""
    if pct is not None and pd.notna(pct):
        p = max(0.0, min(100.0, float(pct)))
        bar = f'<div class="bar"><i style="width:{p:.0f}%"></i></div>'
    is_link = bool(target) and target in NAV_PAGES
    go_html = f'<div class="go">{hint or "Voir le détail"} →</div>' if is_link else ""
    html = (f'<div class="kpi {variant} {"pulse" if pulse else ""} fade-up">'
            f'<div class="label">{ic(icon) if icon else ""}{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div>{bar}{go_html}</div>')
    clickable(html, target, state)


def nav_tile(key: str, state: dict | None = None) -> None:
    """Grande tuile de navigation (accueil) pointant vers la page `key`."""
    meta = PAGE_META[key]
    html = (f'<div class="tile fade-up"><div class="tile-ic">{ICON[meta["svg"]]}</div>'
            f'<div><b>{meta["title"]}</b><span class="d">{meta["desc"]}</span></div>'
            f'<span class="arrow">→</span></div>')
    clickable(html, key, state)


def section(title: str, icon: str | None = None) -> None:
    """Titre de section avec filet vert."""
    st.markdown(f'<div class="section-title fade-up">{ic(icon) if icon else ""}{title}</div>',
                unsafe_allow_html=True)


def box(kind: str, text: str, title: str | None = None) -> None:
    """Encadré coloré. kind : info · tip · alert · ok."""
    icons = {"info": "info", "tip": "info", "alert": "alert", "ok": "check"}
    titles = {"info": "À savoir", "tip": "Astuce", "alert": "Attention", "ok": "Bon à savoir"}
    st.markdown(
        f'<div class="box {kind} fade-up">{ic(icons.get(kind, "info"))}<div>'
        f'<strong>{title or titles.get(kind, "")}</strong> — {text}</div></div>',
        unsafe_allow_html=True)


def badge_classe(classe) -> str:
    """Pastille colorée pour une classe de vulnérabilité."""
    mapping = {"Critique": "badge-critique", "Élevée": "badge-eleve",
               "Modérée": "badge-modere", "Faible": "badge-faible"}
    return (f'<span class="badge {mapping.get(str(classe), "badge-modere")}">'
            f'<span class="dot"></span>{classe}</span>')


def _clean_fig(fig, height=420, source=None, title=None, x_title=None, y_title=None):
    """Applique le style maison à un graphique Plotly (fond blanc, polices…)."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60 if title else 30, b=60 if source else 40),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        font=dict(family="Inter", size=12, color="#14211c"),
        title=dict(text=title, font=dict(family="Sora", size=14, color="#14211c"),
                   x=0.01, xanchor="left") if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0)"),
        xaxis=dict(showgrid=True, gridcolor="#eef1f0", title=x_title),
        yaxis=dict(showgrid=True, gridcolor="#eef1f0", title=y_title),
    )
    if source:
        fig.add_annotation(xref="paper", yref="paper", x=0, y=-0.20, showarrow=False,
                           text=f'<span style="font-size:10px; color:#4b5a54;">Source : {source}</span>')
    return fig


def folium_map(gdf, value_col, legend, height=520, fill_color="YlOrRd", regions_gdf=None):
    """Carte choroplèthe (territoires colorés selon `value_col`).

    📘 Les géométries sont "simplifiées" (moins de points) : la carte est
       beaucoup plus légère et fluide, sans différence visible à l'œil.
    """
    if gdf is None or len(gdf) == 0:
        return folium.Map(location=[8.6, 0.8], zoom_start=6)

    gdf = gdf.copy()
    try:
        tol = 0.01 if len(gdf) < 50 else (0.02 if len(gdf) < 200 else 0.05)
        gdf["geometry"] = gdf.geometry.simplify(tolerance=tol, preserve_topology=True)
    except Exception:
        pass
    gj = __import__("json").loads(gdf.to_json())

    try:
        b = gdf.total_bounds                       # [minx, miny, maxx, maxy]
        centre = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
    except Exception:
        b, centre = None, [8.6, 0.8]

    m = folium.Map(location=centre, zoom_start=7 if len(gdf) <= 50 else 6,
                   tiles="OpenStreetMap", control_scale=True)
    if b is not None and np.all(np.isfinite(b)):
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])      # cadre automatiquement la zone

    # Contours des régions (tirets verts) en fond, si fournis.
    if regions_gdf is not None and len(regions_gdf) > 0:
        folium.GeoJson(
            __import__("json").loads(regions_gdf.to_json()),
            style_function=lambda _: {"fillOpacity": 0, "color": "#00563f",
                                      "weight": 3, "opacity": 0.85, "dashArray": "8 4"},
        ).add_to(m)

    # Couche colorée selon la valeur.
    folium.Choropleth(
        geo_data=gj, data=gdf, columns=["commune", value_col],
        key_on="feature.properties.commune", fill_color=fill_color,
        fill_opacity=0.85, line_opacity=0, nan_fill_color="#e5e7eb",
        legend_name=legend, highlight=True,
    ).add_to(m)

    # Couche interactive : contours + infobulle + surbrillance jaune au survol.
    fields = [f for f in ("commune", "region", value_col) if f in gdf.columns]
    aliases = {"commune": "Territoire", "region": "Région", value_col: legend}
    folium.GeoJson(
        gj,
        style_function=lambda _: {"fillOpacity": 0, "color": "#1a1a1a",
                                  "weight": 1.8, "opacity": 0.75},
        highlight_function=lambda _: {"fillOpacity": 0.25, "fillColor": "#FFCE00",
                                      "color": "#006a4e", "weight": 3.5},
        tooltip=folium.GeoJsonTooltip(fields=fields,
                                      aliases=[aliases[f] for f in fields],
                                      localize=True, sticky=True),
    ).add_to(m)
    return m


def prioritization_map(gdf):
    """Carte des quatre cadrans de la matrice avec les mêmes seuils que le nuage."""
    map_df = gdf.copy()
    map_df["zone_priorisation"] = prioritization_quadrant(
        map_df["score_services"],
        map_df["score_inondation"],
        services_threshold=MATRIX_SERVICES_THRESHOLD,
        flood_threshold=MATRIX_FLOOD_THRESHOLD,
    )
    map_df["geometry"] = map_df.geometry.simplify(
        tolerance=0.01 if len(map_df) < 50 else (0.02 if len(map_df) < 200 else 0.05),
        preserve_topology=True,
    )
    geojson = __import__("json").loads(map_df.to_json())
    bounds = map_df.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=7 if len(map_df) <= 50 else 6,
                   tiles="OpenStreetMap", control_scale=True)
    if np.all(np.isfinite(bounds)):
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    zone_colors = {
        "Urgence": "#C8102E",
        "À équiper": "#d97706",
        "À protéger": "#0284c7",
        "Suivi": "#10b981",
        "Données insuffisantes": "#9ca3af",
    }
    tooltip_fields = [
        col for col in (
            "commune", "region", "zone_priorisation",
            "score_services", "score_inondation",
        )
        if col in map_df.columns
    ]
    tooltip_aliases = {
        "commune": "Territoire",
        "region": "Région",
        "zone_priorisation": "Priorité",
        "score_services": "Déficit de services",
        "score_inondation": "Risque d'inondation",
    }
    folium.GeoJson(
        geojson,
        style_function=lambda feature: {
            "fillColor": zone_colors.get(
                feature["properties"].get("zone_priorisation"), "#9ca3af"
            ),
            "fillOpacity": 0.78,
            "color": "#ffffff",
            "weight": 1,
        },
        highlight_function=lambda _: {
            "fillOpacity": 0.95,
            "color": "#14211c",
            "weight": 2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[tooltip_aliases[col] for col in tooltip_fields],
            localize=True,
            sticky=True,
        ),
    ).add_to(m)
    return m, map_df


def show_map(m, height=560, key=None, clickable_map=False):
    """Affiche une carte Folium. Si `clickable_map`, renvoie le territoire cliqué.

    📘 returned_objects=[] = "ne relance pas le script quand on déplace la carte"
       → beaucoup plus fluide. On ne demande le clic que si nécessaire."""
    kwargs = dict(height=height, key=key,
                  returned_objects=["last_active_drawing"] if clickable_map else [])
    try:
        return st_folium(m, use_container_width=True, **kwargs)
    except TypeError:                                   # anciennes versions
        return st_folium(m, width=1200, **kwargs)


def open_fiche(commune: str, key: str, label: str = "Ouvrir la fiche →") -> None:
    """Bouton : mémorise le territoire et ouvre la Fiche territoriale."""
    if st.button(label, key=key, type="secondary"):
        st.session_state["fiche_target"] = commune
        st.switch_page(NAV_PAGES["fiche"])


def table_with_select(df: pd.DataFrame, key: str, name_col: str, height: int = 440):
    """Tableau où l'on peut cliquer une ligne ; renvoie le nom du territoire choisi.

    📘 Si votre version de Streamlit est ancienne, le tableau s'affiche
       simplement sans sélection (aucune erreur)."""
    try:
        ev = st.dataframe(df, width="stretch", hide_index=True, height=height,
                          on_select="rerun", selection_mode="single-row", key=key)
        rows = ev.selection.rows
        if rows:
            return df.iloc[rows[0]][name_col]
    except Exception:
        st.dataframe(df, width="stretch", hide_index=True, height=height)
    return None


def style_alerts(df, bool_cols):
    """Colore en rouge les cellules True (alertes) d'un tableau."""
    def _hl(v):
        return "background-color:#fde8ea;color:#C8102E;font-weight:600" if v is True or v == True else ""  # noqa: E712
    try:
        return df.style.map(_hl, subset=bool_cols)
    except AttributeError:                              # pandas < 2.1
        return df.style.applymap(_hl, subset=bool_cols)


def bouton_imprimer(label: str = "Imprimer") -> None:
    """Bouton d'impression (ouvre la boîte de dialogue du navigateur)."""
    col_btn, col_hint = st.columns([1, 2])
    with col_btn:
        if st.button(f"🖨 {label}", type="primary", width="stretch", key=f"print_{label}"):
            # Un mini-script invisible déclenche window.print() dans la page parente.
            components.html("<script>setTimeout(function(){try{window.parent.print();}"
                            "catch(e){window.print();}},250);</script>", height=0)
    with col_hint:
        st.caption("💡 Ou **Ctrl+P** / **Cmd+P** — la barre latérale est masquée à l'impression.")


# ── Filtres rapides ("focus") posés par les KPI cliquables ──────────────────
FOCUS_LABEL = {
    "critique": "Territoires critiques (ICV > 75)",
    "double_peine": "Double peine (services + inondation)",
    "alerte_sante": "Alerte Santé (norme OMS)",
    "alerte_eau": "Alerte Eau (norme Sphère)",
    "alerte_alim": "Alerte Alimentation",
    "alerte_ino": "Alerte Inondation",
    "alerte_ero": "Alerte Érosion",
}


def apply_focus(df: pd.DataFrame, focus: str | None) -> pd.DataFrame:
    """Réduit `df` selon le filtre mémorisé par un KPI cliqué."""
    if not focus or df.empty:
        return df
    if focus == "critique" and "classe" in df.columns:
        return df[df["classe"] == "Critique"]
    if focus in df.columns:                           # double_peine, alerte_*
        return df[df[focus].fillna(False).astype(bool)]
    return df


def metric_label(p: str) -> str:
    """Libellé du sélecteur de carte pour un pilier (ex. 'Pilier Eau')."""
    return f"Pilier {PILLAR_LABEL[p]}"


# ════════════════════════════════════════════════════════════════
# 5. BARRE LATÉRALE : niveau, filtres, pondération
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    # (Le logo est posé en haut par st.logo ; le menu vient juste dessous.)
    st.markdown(f'<div class="side-h">{ic("map")}<span>Maille territoriale</span></div>',
                unsafe_allow_html=True)
    niveau_label = st.radio("Niveau", list(NIVEAUX), index=2,
                            label_visibility="collapsed", key="side_niveau",
                            help="Choisissez l'échelle d'analyse : de la région au canton.")
    niveau = NIVEAUX[niveau_label]
    gdf_base = levels.get(niveau, pd.DataFrame())

    if is_demo:
        box("alert", "Mode démonstration actif : les chiffres sont fictifs.")

    st.markdown(f'<div class="side-h">{ic("filter")}<span>Filtres</span></div>',
                unsafe_allow_html=True)
    region_sel = "Toutes"
    if niveau != "regions" and len(gdf_base) and "region" in gdf_base.columns:
        regions = ["Toutes"] + sorted(r for r in gdf_base["region"].dropna().unique()
                                      if r != "N/D")
        region_sel = st.selectbox("Région", regions, key="side_region")

    mode_citoyen = st.toggle("Mode citoyen", value=False, key="side_citoyen",
                             help="Langage simple + fiche imprimable dans « Fiche territoriale ».")

    st.markdown(f'<div class="side-h">{ic("scale")}<span>Pondération ICV</span></div>',
                unsafe_allow_html=True)

    def _reset_weights():
        """Remet les 4 curseurs à 0,25 (appelé au clic sur « Réinitialiser »)."""
        for p_ in PILLARS:
            st.session_state[f"w_{niveau}_{p_}"] = 0.25

    with st.expander("Ajuster les poids", expanded=False):
        w = {p: st.slider(PILLAR_LABEL[p], 0.0, 1.0, 0.25, 0.05, key=f"w_{niveau}_{p}")
             for p in PILLARS}
        total = sum(w.values()) or 1.0
        w = {k: v / total for k, v in w.items()}        # on normalise : somme = 1
        st.caption(f"Somme = {total:.2f} → normalisée à 1.00")
        st.button("↺ Réinitialiser", on_click=_reset_weights, key="reset_w")

    with st.expander("❓ Aide rapide"):
        st.markdown(
            "- **Cartes KPI & tuiles** : cliquez dessus, elles ouvrent la page liée.\n"
            "- **Carte / graphique / tableau** : cliquez un territoire pour ouvrir sa fiche.\n"
            "- **Poids** : donnez plus d'importance à un pilier pour recalculer l'ICV.")

    st.markdown("---")
    st.caption(f"**{len(gdf_base)} {niveau_label.lower()}**\n\n"
               "**Sources** : opendata.gouv.tg · geodata.gouv.tg")

# Calcul de l'ICV avec les poids choisis (résultat mis en cache).
if len(gdf_base):
    try:
        gdf = compute_icv_cached(niveau, tuple(sorted(w.items())), DATA_VERSION)
        for col in gdf_base.columns:                    # sécurité : aucune colonne perdue
            if col not in gdf.columns:
                gdf[col] = gdf_base[col].values
    except Exception:
        gdf = compute_icv(gdf_base.copy(), w)
else:
    gdf = gdf_base

# `gdf_view` = les données après filtre « Région » de la barre latérale.
if len(gdf) and region_sel != "Toutes" and "region" in gdf.columns:
    gdf_view = gdf[gdf["region"] == region_sel].copy()
else:
    gdf_view = gdf

regions_overlay = levels.get("regions", pd.DataFrame())


# ════════════════════════════════════════════════════════════════
# 6. LES PAGES  (une fonction = une page)
# ════════════════════════════════════════════════════════════════
# 📘 Pour ajouter une page : 1) écrivez une fonction page_xxx()
#    2) ajoutez-la dans PAGES_DEF au §7. C'est tout.

# ─────────────────────────── ACCUEIL ───────────────────────────
def page_accueil():
    hero('VIGIE<span class="dot">·</span>TOGO 360',
         f"{APP_TAGLINE}. Cliquez sur une tuile ou un indicateur pour explorer.",
         chips=["4 piliers ICV", "4 niveaux", f"{len(gdf_view)} {niveau_label.lower()}",
                "Aide à la décision"])
    if gdf_view.empty:
        box("alert", "Aucune donnée pour ce niveau. Vérifiez le dossier data/.")
        return

    n_crit = int((gdf_view["classe"] == "Critique").sum()) if "classe" in gdf_view else 0
    n_dp = int(gdf_view["double_peine"].fillna(False).sum()) if "double_peine" in gdf_view else 0
    icv_moy = float(gdf_view["ICV"].mean())

    # 1) Bannière d'alerte cliquable (seulement s'il y a des territoires critiques)
    if n_crit > 0:
        clickable(
            f'<div class="banner fade-up">{ic("alert")}<div><b>{n_crit} territoire(s) en situation '
            f'critique</b> parmi les {niveau_label.lower()} affichés.<br>'
            f'<span style="font-size:.8rem">Ils cumulent les plus fortes vulnérabilités (ICV &gt; 75).</span>'
            f'</div><span class="arrow">Voir la liste →</span></div>',
            "dashboard", {"focus": "critique"})

    # 2) Chiffres clés (chaque KPI ouvre une page)
    section("Chiffres clés", "dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Territoires", len(gdf_view), niveau_label.lower(), icon="map", pct=100,
                 target="dashboard", state={"focus": None}, hint="Ouvrir la carte")
    with c2:
        kpi_card("Critiques", n_crit, "ICV > 75", "danger", icon="alert", pulse=n_crit > 0,
                 pct=100 * n_crit / max(len(gdf_view), 1),
                 target="dashboard", state={"focus": "critique"}, hint="Voir les critiques")
    with c3:
        kpi_card("Double peine", n_dp, "services + inondation", "warn", icon="alert",
                 pct=100 * n_dp / max(len(gdf_view), 1),
                 target="analyse", hint="Analyser les hot-spots")
    with c4:
        kpi_card("ICV moyen", f"{icv_moy:.1f}", "0 – 100", icon="scale", pct=icv_moy,
                 target="scenarios", hint="Simuler une amélioration")

    # 3) Tuiles de navigation guidée
    section("Que voulez-vous faire ?", "analyse")
    for row in (["dashboard", "analyse", "scenarios", "decision"],
                ["fiche", "education", "catalogue", "projets"],
                ["normes", "methode", "qualite", "sources"]):
        cols = st.columns(4)
        for col, key in zip(cols, row):
            with col:
                nav_tile(key)

    # 4) Les 4 piliers ICV (chacun ouvre la carte colorée selon ce pilier)
    section("Les 4 piliers ICV — cliquez pour les voir sur la carte", "analyse")
    cols = st.columns(4)
    for i, p in enumerate(PILLARS):
        with cols[i]:
            html = (f'<div class="kpi fade-up" style="border-top-color:{PILLAR_COLOR[p]}">'
                    f'<div style="font-size:1.6rem;color:{PILLAR_COLOR[p]};margin-bottom:.4rem">{ICON[p]}</div>'
                    f'<div style="font-weight:700;margin-bottom:.25rem">{PILLAR_LABEL[p]}</div>'
                    f'<div class="sub">{PILLAR_DESC[p]}</div>'
                    f'<div class="go" style="color:{PILLAR_COLOR[p]}">Voir la carte →</div></div>')
            clickable(html, "dashboard", {"map_metric_target": metric_label(p), "focus": None})

    # 5) Top 5 + répartition par classe
    left, right = st.columns([3, 2])
    with left:
        section("Les 5 territoires les plus vulnérables", "alert")
        top = gdf_view.sort_values("ICV", ascending=False).head(5)
        for i, (_, r) in enumerate(top.iterrows()):
            a, b, c = st.columns([4, 2, 2])
            a.markdown(f"**{i + 1}. {r['commune']}**<br><small>{r.get('region', '')}</small>",
                       unsafe_allow_html=True)
            b.markdown(f"ICV **{r['ICV']:.0f}** &nbsp;{badge_classe(r.get('classe', ''))}",
                       unsafe_allow_html=True)
            with c:
                open_fiche(r["commune"], key=f"home_top_{i}", label="Fiche →")
    with right:
        section("Répartition par classe", "analyse")
        if "classe" in gdf_view.columns:
            cnt = gdf_view["classe"].value_counts().reindex(
                ["Critique", "Élevée", "Modérée", "Faible"]).dropna().reset_index()
            cnt.columns = ["classe", "n"]
            fig = px.pie(cnt, names="classe", values="n", hole=0.58, color="classe",
                         color_discrete_map=CLASS_COLOR)
            fig.update_traces(textinfo="value+percent", sort=False)
            st.plotly_chart(_clean_fig(fig, 330), width="stretch")


# ─────────────────────── TABLEAU DE BORD ───────────────────────
def page_dashboard():
    hero(f"Tableau de bord — {niveau_label}",
         "Indice Communal de Vulnérabilité · 100 = plus vulnérable",
         chips=[f"{len(gdf_view)} territoires", f"Région : {region_sel}"])
    if gdf_view.empty:
        box("alert", "Aucune donnée.")
        return

    # --- KPI (calculés sur TOUS les territoires visibles, avant filtre rapide)
    n_crit = int((gdf_view["classe"] == "Critique").sum()) if "classe" in gdf_view else 0
    n_dp = int(gdf_view["double_peine"].fillna(False).sum()) if "double_peine" in gdf_view else 0
    icv_moy = gdf_view["ICV"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Territoires", len(gdf_view), niveau_label.lower(), icon="map", pct=100,
                 target="dashboard", state={"focus": None}, hint="Tout afficher")
    with c2:
        kpi_card("Critiques", n_crit, "ICV > 75", "danger", icon="alert", pulse=n_crit > 0,
                 pct=100 * n_crit / max(len(gdf_view), 1),
                 target="dashboard", state={"focus": "critique"}, hint="Filtrer les critiques")
    with c3:
        kpi_card("Double peine", n_dp, "services + inondation", "warn", icon="alert",
                 pct=100 * n_dp / max(len(gdf_view), 1),
                 target="dashboard", state={"focus": "double_peine"}, hint="Filtrer")
    with c4:
        kpi_card("ICV moyen", f"{icv_moy:.1f}", "0 – 100", icon="scale", pct=icv_moy,
                 target="scenarios", hint="Simuler")

    # --- Filtre rapide actif (posé par un KPI cliqué)
    focus = st.session_state.get("focus")
    df_f = apply_focus(gdf_view, focus)
    if focus:
        a, b = st.columns([5, 1])
        with a:
            box("info", f"<b>{FOCUS_LABEL.get(focus, focus)}</b> — {len(df_f)} territoire(s) affiché(s).",
                "Filtre actif")
        with b:
            if st.button("✕ Retirer", key="clear_focus", width="stretch"):
                st.session_state["focus"] = None
                st.rerun()
        if df_f.empty:
            box("ok", "Aucun territoire ne correspond à ce filtre.")
            return

    # --- Indicateur affiché (ICV ou un pilier)
    options = {"ICV global": "ICV"}
    for p in PILLARS:
        if f"score_{p}" in df_f.columns:
            options[metric_label(p)] = f"score_{p}"
    labels = list(options)
    tgt = st.session_state.pop("map_metric_target", None)   # posé par une tuile "pilier"
    if tgt in labels:
        st.session_state["map_metric_sel"] = tgt
    sel_label = st.selectbox("Indicateur affiché", labels, key="map_metric_sel",
                             help="Colorie la carte, le classement et la distribution.")
    mcol = options[sel_label]

    tab1, tab2, tab3, tab4 = st.tabs(["🗺 Carte", "🏆 Top 15", "📊 Distribution", "📋 Tableau"])

    with tab1:
        st.caption("👆 Cliquez sur un territoire pour proposer d'ouvrir sa fiche.")
        m = folium_map(df_f, mcol, sel_label, regions_gdf=regions_overlay)
        out = show_map(m, 560, key="main_map", clickable_map=True)
        feat = (out or {}).get("last_active_drawing") or {}
        picked = (feat.get("properties") or {}).get("commune")
        if picked:
            a, b = st.columns([4, 1])
            with a:
                box("tip", f"Territoire sélectionné : <b>{picked}</b>", "Sélection")
            with b:
                open_fiche(picked, key="map_open_fiche")

    with tab2:
        top = df_f.sort_values(mcol, ascending=False).head(15)
        fig = px.bar(top, x=mcol, y="commune", orientation="h", color=mcol,
                     color_continuous_scale="YlOrRd", range_color=[0, 100],
                     labels={mcol: sel_label, "commune": ""}, text=mcol)
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig = _clean_fig(fig, 500, "opendata.gouv.tg", title=f"Top 15 — {niveau_label}")
        st.caption("👆 Cliquez sur une barre pour ouvrir la fiche du territoire.")
        try:
            ev = st.plotly_chart(fig, width="stretch", key="top15_chart",
                                 on_select="rerun", selection_mode="points")
            pts = ev.selection.points
            if pts:
                nm = pts[0].get("y")
                a, b = st.columns([4, 1])
                with a:
                    box("tip", f"Barre sélectionnée : <b>{nm}</b>", "Sélection")
                with b:
                    open_fiche(nm, key="top15_open_fiche")
        except Exception:
            st.plotly_chart(fig, width="stretch")

    with tab3:
        fig = px.histogram(df_f, x=mcol, nbins=20, color_discrete_sequence=[C_GREEN],
                           labels={mcol: f"{sel_label} (0–100)"})
        st.plotly_chart(_clean_fig(fig, 400, "calcul VIGIE·TOGO",
                                   title=f"Distribution — {niveau_label}",
                                   y_title="Nb de territoires"), width="stretch")

    with tab4:
        st.caption("👆 Cliquez une ligne pour ouvrir la fiche du territoire.")
        cols_t = [c for c in ["commune", "region", "ICV", "classe"] +
                  [f"score_{p}" for p in PILLARS] if c in df_f.columns]
        dft = df_f[cols_t].sort_values("ICV", ascending=False).round(1).rename(
            columns={"commune": "Territoire", "region": "Région", "classe": "Classe"})
        nm = table_with_select(dft, "dash_table", "Territoire")
        if nm:
            open_fiche(nm, key="table_open_fiche", label=f"Ouvrir la fiche de {nm} →")

    st.download_button(
        "⬇ Télécharger CSV",
        data=df_f.drop(columns="geometry", errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name=f"icv_{niveau}.csv", mime="text/csv")


# ─────────────────────── ANALYSE TERRITORIALE ───────────────────────
def page_analyse():
    hero("Analyse territoriale", "Matrice de priorisation, corrélations et hot-spots.",
         chips=["Matrice 4 cadrans", "Hot-spots"])
    if gdf_view.empty:
        return

    section("Matrice de priorisation", "analyse")
    box("tip", "Choisissez la carte pour repérer les zones prioritaires par territoire, "
        "ou la matrice à bulles pour comparer les deux scores.")
    if "score_services" in gdf_view.columns and "score_inondation" in gdf_view.columns:
        plot_df = gdf_view.dropna(subset=["score_services", "score_inondation"]).copy()
        matrix_view = st.radio(
            "Mode de lecture",
            ["Carte du Togo", "Matrice à bulles"],
            horizontal=True,
            key="prioritization_view",
        )
        if matrix_view == "Carte du Togo":
            st.caption(
                "Les couleurs reprennent les seuils graphiques de la matrice "
                f"(services ≥ {MATRIX_SERVICES_THRESHOLD}, inondation ≥ "
                f"{MATRIX_FLOOD_THRESHOLD}). Ils sont distincts du scénario "
                "analytique de double vulnérabilité 60/30."
            )
            st.caption(
                "Légende : 🔴 Urgence · 🟠 À équiper · 🔵 À protéger · "
                "🟢 Suivi · ⚪ Données insuffisantes. Survolez une zone pour voir "
                "son nom et ses scores."
            )
            priority_map, priority_df = prioritization_map(gdf_view)
            show_map(priority_map, height=540, key="prioritization_map")

            zone_order = [
                "Urgence", "À équiper", "À protéger", "Suivi",
                "Données insuffisantes",
            ]
            zone_counts = priority_df["zone_priorisation"].value_counts()
            for col, zone in zip(st.columns(len(zone_order)), zone_order):
                col.metric(zone, int(zone_counts.get(zone, 0)))

            urgent = priority_df[
                priority_df["zone_priorisation"] == "Urgence"
            ]
            if not urgent.empty:
                section("Territoires en zone d'urgence", "alert")
                columns = [
                    col for col in ("commune", "region", "score_services", "score_inondation")
                    if col in urgent.columns
                ]
                urgent = urgent[columns].sort_values(
                    ["score_inondation", "score_services"], ascending=False
                ).rename(columns={
                    "commune": "Territoire",
                    "region": "Région",
                    "score_services": "Déficit de services",
                    "score_inondation": "Risque d'inondation",
                })
                st.dataframe(urgent.round(1), width="stretch", hide_index=True)
        else:
            size_col = None
            if "population" in plot_df.columns:     # taille de bulle = population (min. 1000)
                plot_df["_pop"] = (pd.to_numeric(plot_df["population"], errors="coerce")
                                   .fillna(1000).clip(lower=1000))
                size_col = "_pop"
            fig = px.scatter(plot_df, x="score_services", y="score_inondation", size=size_col,
                             color="ICV", hover_name="commune", custom_data=["commune"],
                             color_continuous_scale="YlOrRd", range_color=[0, 100],
                             labels={"score_services": "Déficit de services",
                                     "score_inondation": "Exposition inondation"})
            fig.add_hline(y=MATRIX_FLOOD_THRESHOLD, line_dash="dash", line_color="gray")
            fig.add_vline(x=MATRIX_SERVICES_THRESHOLD, line_dash="dash", line_color="gray")
            for x, y, txt, c in [(25, 25, "Stable", "#059669"), (82, 25, "À équiper", "#d97706"),
                                 (25, 82, "À protéger", "#0284c7"), (82, 82, "URGENCE", C_RED)]:
                fig.add_annotation(x=x, y=y, text=f"<b>{txt}</b>", showarrow=False,
                                   font=dict(color=c, size=14))
            fig = _clean_fig(fig, 520, "ISRI-TG", title="Matrice de priorisation")
            try:
                ev = st.plotly_chart(fig, width="stretch", key="matrix_chart",
                                     on_select="rerun", selection_mode="points")
                pts = ev.selection.points
                if pts and pts[0].get("customdata"):
                    nm = pts[0]["customdata"][0]
                    a, b = st.columns([4, 1])
                    with a:
                        box("tip", f"Bulle sélectionnée : <b>{nm}</b>", "Sélection")
                    with b:
                        open_fiche(nm, key="matrix_open_fiche")
            except Exception:
                st.plotly_chart(fig, width="stretch")

    section("Hot-spots (3 alertes ou plus)", "alert")
    ap = [c for c in ["alerte_sante", "alerte_eau", "alerte_alim", "alerte_ino", "alerte_ero"]
          if c in gdf_view.columns]
    if ap:
        dfv = gdf_view.copy()
        dfv["_n_alertes"] = dfv[ap].fillna(False).astype(int).sum(axis=1)
        hot = dfv[dfv["_n_alertes"] >= 3].sort_values(["_n_alertes", "ICV"], ascending=False)
        if len(hot):
            box("alert", f"<strong>{len(hot)}</strong> territoires cumulent 3 alertes ou plus.")
            show = [c for c in ["commune", "region", "_n_alertes", "ICV"] if c in hot.columns]
            dfh = hot[show].round(1).rename(columns={"commune": "Territoire", "region": "Région",
                                                     "_n_alertes": "Alertes"})
            nm = table_with_select(dfh, "hot_table", "Territoire", height=360)
            if nm:
                open_fiche(nm, key="hot_open_fiche", label=f"Ouvrir la fiche de {nm} →")
        else:
            box("ok", "Aucun territoire ne cumule 3 alertes.")


# ─────────────────────────── SCÉNARIOS ───────────────────────────
def page_scenarios():
    hero("Scénarios", "Simulez l'impact d'une amélioration sur l'indice.",
         chips=["Simulation", "Avant / après"])
    if gdf_view.empty:
        return

    section("Paramètres", "scenario")
    box("info", "Choisissez un ou plusieurs piliers et un pourcentage de réduction du "
        "déficit : l'application recalcule l'ICV de chaque territoire.")
    c1, c2 = st.columns([2, 1])
    with c1:
        piliers_cibles = st.multiselect("Piliers à améliorer", PILLARS, default=["eau"],
                                        format_func=lambda p: PILLAR_LABEL[p])
    with c2:
        reduction_pct = st.slider("Réduction (%)", 0, 100, 50, 5)
    if not piliers_cibles:
        box("tip", "Sélectionnez au moins un pilier pour lancer la simulation.")
        return

    # On réduit le score des piliers ciblés, puis on recalcule l'ICV.
    gdf_s = gdf_view.copy()
    for p in piliers_cibles:
        cs = f"score_{p}"
        if cs in gdf_s.columns:
            gdf_s[cs] = gdf_s[cs] * (1 - reduction_pct / 100)
    icv_s = compute_icv(gdf_s, w)

    n_a = int((gdf_view["classe"] == "Critique").sum())
    n_b = int((icv_s["classe"] == "Critique").sum())
    moy_a, moy_b = gdf_view["ICV"].mean(), icv_s["ICV"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        d = n_b - n_a
        kpi_card("Critiques", f"{n_a} → {n_b}", f"{d:+d}", "ok" if d < 0 else "danger" if d > 0 else "",
                 icon="alert", target="dashboard", state={"focus": "critique"},
                 hint="Voir les critiques actuels")
    with c2:
        d = moy_b - moy_a
        kpi_card("ICV moyen", f"{moy_a:.1f} → {moy_b:.1f}", f"{d:+.2f}", "ok" if d < 0 else "",
                 icon="scale")
    with c3:
        kpi_card("Piliers améliorés", len(piliers_cibles), f"à −{reduction_pct} %", icon="scenario")

    section("Distribution avant / après", "prix")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=gdf_view["ICV"].dropna(), name="Avant",
                               marker_color="#94a3b8", opacity=0.7, nbinsx=25))
    fig.add_trace(go.Histogram(x=icv_s["ICV"].dropna(), name="Après",
                               marker_color=C_GREEN, opacity=0.75, nbinsx=25))
    fig.update_layout(barmode="overlay")
    st.plotly_chart(_clean_fig(fig, 400, "simulation", title="Impact du scénario",
                               x_title="ICV (0–100)", y_title="Nb de territoires"),
                    width="stretch")


# ─────────────────────── PROJETS & INVESTISSEMENTS ───────────────────────

    st.subheader("📊 Analyse de sensibilité — Seuils de double peine")

    from icv import double_peine_scenarios
    # ── Récupération des scénarios ────────────────────────────────
    seuils_df = double_peine_scenarios(gdf_view)

    # ── Création d'un libellé catégoriel pour l'axe X ─────────────
    seuils_df["scenario"] = (
        seuils_df["seuil_services"].astype(int).astype(str)
        + " / " + seuils_df["seuil_inondation"].astype(int).astype(str)
    )

    # ── Graphique corrigé ─────────────────────────────────────────
    fig = px.bar(
        seuils_df,
        x="scenario",                           # ← catégoriel au lieu de seuil_services
        y="communes_identifiees",
        text="communes_identifiees",
        color="communes_identifiees",           # ← on garde le dégradé de couleur
        color_continuous_scale="YlOrRd",
        labels={"scenario": "Seuils (services / inondation)",
                "communes_identifiees": "Communes identifiées"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,                       # ← supprime la colorbar
        coloraxis_showscale=False,              # ← supprime aussi la barre de dégradé
        margin=dict(t=60, b=40, l=20, r=20),    # ← évite que le titre soit coupé
        xaxis=dict(type="category"),            # ← force l'axe X en catégories
    )
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────────────────────────
    # 4. MÉTHODOLOGIE & GUIDE COMPLET

def page_projets():
    hero("Projets & investissements", "Suivi des chantiers, taux de réalisation, ZAAP.",
         chips=["COSO", "ZAAP"])
    if gdf_view.empty:
        return

    has_p = "pciaeph_total" in gdf_view.columns
    has_z = "zaap_nb" in gdf_view.columns
    if not has_p and not has_z:
        box("alert", "Données de projets non disponibles. Sélectionnez le niveau <b>Communes</b>.")
        return

    zero = pd.Series(0, index=gdf_view.index)          # série de zéros si colonne absente
    if has_p:
        section("Projets hydrauliques COSO", "projets")
        total = int(gdf_view["pciaeph_total"].fillna(0).sum())
        acheves = int((gdf_view.get("pciaeph_acheve", zero).fillna(0)
                       + gdf_view.get("pciaeph_receptionne_prov", zero).fillna(0)
                       + gdf_view.get("pciaeph_receptionne_def", zero).fillna(0)).sum())
        encours = int(gdf_view.get("pciaeph_en_cours", zero).fillna(0).sum())
        cout = gdf_view.get("pciaeph_cout", zero).fillna(0).sum()
        taux = acheves / max(total, 1)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Projets", total, "COSO", icon="projets", pct=100,
                     target="catalogue", hint="Ouvrir le catalogue")
        with c2:
            kpi_card("Achevés", acheves, f"{100 * taux:.0f} %", "ok" if taux > .6 else "warn",
                     icon="check", pct=100 * taux, target="catalogue", hint="Voir le détail")
        with c3:
            kpi_card("En cours", encours, "actifs", "warn", icon="projets",
                     target="catalogue", hint="Voir le détail")
        with c4:
            kpi_card("Budget", f"{cout / 1e6:.1f} M FCFA" if cout > 0 else "N/D", icon="scale",
                     target="catalogue", hint="Voir le détail")

    if has_z:
        section("Zones agricoles (ZAAP)", "alimentation")
        nbz = int(gdf_view["zaap_nb"].fillna(0).sum())
        surf = gdf_view.get("zaap_surface_ha", zero).fillna(0).sum()
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("ZAAP", nbz, "zones", icon="alimentation")
        with c2:
            kpi_card("Surface", f"{surf:,.0f} ha".replace(",", " "), "protégée", icon="map")


# ─────────────────────── DENRÉES & MARCHÉS ───────────────────────
def page_denrees():
    """Page déléguée au fichier pages_denrees.py (même dossier)."""
    try:
        from pages_denrees import render as render_denrees
        render_denrees(hero, kpi_card, section, box, folium_map, _clean_fig,
                       levels, prices, levels.get("communes", pd.DataFrame()))
    except ImportError:
        box("alert", "Le fichier <code>pages_denrees.py</code> est introuvable. "
            "Créez-le à la racine du projet (même dossier que app.py).")
    except Exception as e:                              # on affiche l'erreur sans planter
        box("alert", f"Erreur sur la page Denrées : {e}")


# ─────────────────────────── ÉDUCATION ───────────────────────────
# 📘 POUR LES DÉBUTANTS : cette page combine
#    (a) des chiffres NATIONAUX officiels, écrits ici sous forme de constantes
#        (EDU_KPI_NATIONAL) : c'est la "source de vérité" de la page ;
#    (b) des données par territoire, lues dans `levels` si elles existent.
#    Si un fichier key_numbers.json est trouvé, ses valeurs remplacent les
#    constantes (pratique pour mettre à jour sans toucher au code).
import json as _json
import os as _os

# ════════════════════════════════════════════════════════════════
# KPI ÉDUCATION NATIONAUX — source de vérité
# ════════════════════════════════════════════════════════════════
EDU_KPI_NATIONAL = {
    "population_totale":        {"value": 8095498, "annee": 2022, "unite": "habitants",      "label": "Population"},
    "etablissements_geo":       {"value": 15454,   "annee": 2024, "unite": "établissements", "label": "Établissements géolocalisés"},
    "batiments_scolaires":      {"value": 28055,   "annee": 2024, "unite": "bâtiments",      "label": "Bâtiments scolaires"},
    "eleves_primaire":          {"value": 1672475, "annee": 2022, "unite": "élèves",         "label": "Élèves au primaire"},
    "enseignants_primaire":     {"value": 41132,   "annee": 2022, "unite": "enseignants",    "label": "Enseignants primaire"},
    "ratio_pe_primaire":        {"value": 40.7,    "annee": 2022, "unite": "élèves/ens.", "norme": 40, "alerte": True, "label": "Ratio P/E primaire"},
    "tbs_prescolaire":          {"value": 37.3,    "annee": 2019, "unite": "%",              "label": "TBS préscolaire"},
    "tbs_primaire":             {"value": 95.0,    "annee": 2022, "unite": "%",              "label": "TBS primaire"},
    "tbs_college":              {"value": 77.1,    "annee": 2022, "unite": "%",              "label": "TBS collège"},
    "tbs_lycee":                {"value": 34.5,    "annee": 2022, "unite": "%", "alerte": True, "label": "TBS lycée"},
    "ips_primaire":             {"value": 0.96,    "annee": 2022, "unite": "ratio F/G", "norme": 1.0, "label": "IPS primaire"},
    "ips_college":              {"value": 0.90,    "annee": 2022, "unite": "ratio F/G", "norme": 1.0, "alerte": True, "label": "IPS collège"},
    "ips_lycee":                {"value": 0.71,    "annee": 2022, "unite": "ratio F/G", "norme": 1.0, "alerte": True, "label": "IPS lycée"},
    "electrification_primaire": {"value": 38.3,    "annee": 2024, "unite": "%",              "label": "Électrification primaire"},
    "electrification_college":  {"value": 67.2,    "annee": 2024, "unite": "%",              "label": "Électrification collège"},
    "electrification_lycee":    {"value": 87.5,    "annee": 2024, "unite": "%",              "label": "Électrification lycée"},
    "eau_primaire":             {"value": 58.5,    "annee": 2024, "unite": "%",              "label": "Accès eau primaire"},
    "internet_primaire":        {"value": 1.7,     "annee": 2024, "unite": "%", "alerte": True, "label": "Internet primaire"},
}

# Points chauds IPS lycée (préfectures les plus inégalitaires)
EDU_IPS_HOTSPOTS = pd.DataFrame([
    {"Préfecture": "Dankpen", "IPS lycée (F/G)": 0.27},
    {"Préfecture": "Vo",      "IPS lycée (F/G)": 0.56},
    {"Préfecture": "Anié",    "IPS lycée (F/G)": 0.59},
])

# Indice composite de risque de décrochage par région (0-100).
# ⚠️ Seules les valeurs Savanes (74,3) et Plateaux (73,8) sont documentées comme
#    vérifiées. Les 3 autres sont marquées "À confirmer" et signalées à l'écran.
EDU_RISQUE_REGIONS = pd.DataFrame([
    {"Région": "Savanes",  "Risque décrochage (0-100)": 74.3, "Statut": "Vérifié"},
    {"Région": "Plateaux", "Risque décrochage (0-100)": 73.8, "Statut": "Vérifié"},
    {"Région": "Kara",     "Risque décrochage (0-100)": 62.0, "Statut": "À confirmer"},
    {"Région": "Centrale", "Risque décrochage (0-100)": 55.0, "Statut": "À confirmer"},
    {"Région": "Maritime", "Risque décrochage (0-100)": 38.0, "Statut": "À confirmer"},
])
EDU_RISQUE_TOP_PREF = ["Dankpen", "Moyen-Mono", "Tandjoaré"]

# Ancres VÉRIFIÉES de la simulation : (cible P/E, enseignants à redéployer, élèves gagnants)
EDU_SIM_ANCHORS = [(35, 790, 209302), (40, 1381, 592549), (50, 66, 59951)]
EDU_SIM_ORIGINE = pd.DataFrame([
    {"Inspection d'origine": "Lomé-Aéroport",  "Enseignants redéployables": 239},
    {"Inspection d'origine": "Lomé-Centre",    "Enseignants redéployables": 154},
    {"Inspection d'origine": "Sotouboua-Nord", "Enseignants redéployables": 132},
])


def load_education_key_numbers(path_candidates=None):
    """Cherche key_numbers.json (à côté d'app.py ou dans le dossier courant).

    Retourne (données, chemin) ou (None, None) : jamais de FileNotFoundError."""
    base = Path(__file__).parent
    rel = ["outputs/key_numbers.json", "../outputs/key_numbers.json",
           "../../outputs/key_numbers.json", "data/key_numbers.json",
           "data_clean/key_numbers.json"]
    for p in (path_candidates or ([str(base / r) for r in rel] + rel)):
        try:
            if _os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return _json.load(f), p
        except Exception:
            continue
    return None, None


def _edu_apply_json(kpis):
    """Met à jour les constantes avec le JSON, clé par clé, si les valeurs sont numériques."""
    data, path = load_education_key_numbers()
    if not isinstance(data, dict):
        return kpis, None
    merged, n = {k: dict(v) for k, v in kpis.items()}, 0
    for k, v in data.items():
        if k not in merged:
            continue
        val = v.get("value") if isinstance(v, dict) else v
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            merged[k]["value"] = val
            if isinstance(v, dict) and v.get("annee"):
                merged[k]["annee"] = v["annee"]
            n += 1
    return merged, ((path, n) if n else None)


EDU_KPI_NATIONAL, EDU_JSON_INFO = _edu_apply_json(EDU_KPI_NATIONAL)


def _edu_kpi(key):
    return EDU_KPI_NATIONAL.get(key, {})


def _edu_v(key, default=0.0):
    """Valeur numérique brute d'un KPI (pour graphiques et barres)."""
    return float(_edu_kpi(key).get("value", default))


def _edu_fmt(key, kind="value"):
    """Formate un KPI éducation en toute sécurité ("N/D" si absent)."""
    k = _edu_kpi(key)
    if not k:
        return "N/D"
    v = k["value"]
    if kind == "int":
        return f"{int(v):,}".replace(",", " ")
    if kind == "pct":
        return f"{v:.1f} %"
    if kind == "ratio":
        return f"{v:.2f}"
    if kind == "annee":
        return str(k.get("annee", "N/D"))
    return v


def _edu_interp_sim(cible):
    """Interpolation linéaire entre les 3 ancres vérifiées (35, 40, 50).

    ⚠️ Seules les cibles 35, 40 et 50 sont exactes ; entre deux ancres, c'est une estimation."""
    a = EDU_SIM_ANCHORS
    if cible <= a[0][0]:
        return a[0][1], a[0][2]
    if cible >= a[-1][0]:
        return a[-1][1], a[-1][2]
    for i in range(len(a) - 1):
        x0, x1 = a[i][0], a[i + 1][0]
        if x0 <= cible <= x1:
            t = (cible - x0) / (x1 - x0)
            return (int(round(a[i][1] + t * (a[i + 1][1] - a[i][1]))),
                    int(round(a[i][2] + t * (a[i + 1][2] - a[i][2]))))
    return 0, 0


def _nom_territoire(row):
    """Nom d'un territoire, sans KeyError si la colonne 'commune' est absente."""
    for col in ("commune", "region", "territoire", "nom"):
        if col in row.index and pd.notna(row.get(col)):
            return str(row[col])
    return "Territoire"


def _num_col(df, col):
    """Colonne numérique d'un DataFrame, ou série vide de NaN si absente."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def page_education():
    """Éducation — 3 défis stratégiques · données DIES / RSA / RGPH 2019-2024."""
    hero("Éducation — 3 défis stratégiques",
         "Décrochage · Inégalités Filles/Garçons · Répartition écoles & enseignants",
         chips=["18 KPI officiels", "2019-2024", "DIES · RSA · RGPH", "Simulation redéploiement"])
    if EDU_JSON_INFO:
        st.caption(f"ℹ️ {EDU_JSON_INFO[1]} valeur(s) mise(s) à jour depuis `{EDU_JSON_INFO[0]}`.")

    communes_df = levels.get("communes", pd.DataFrame()) if isinstance(levels, dict) else pd.DataFrame()
    regions_df = levels.get("regions", pd.DataFrame()) if isinstance(levels, dict) else pd.DataFrame()

    # ── Bandeau 1 : structure du système ─────────────────────────
    section("Structure du système éducatif", "dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Population", _edu_fmt("population_totale", "int"),
                 f"habitants · {_edu_fmt('population_totale', 'annee')}", icon="pop", pct=100)
    with c2:
        kpi_card("Établissements", _edu_fmt("etablissements_geo", "int"),
                 f"géolocalisés · {_edu_fmt('etablissements_geo', 'annee')}", icon="fiche", pct=100)
    with c3:
        kpi_card("Bâtiments", _edu_fmt("batiments_scolaires", "int"),
                 f"scolaires · {_edu_fmt('batiments_scolaires', 'annee')}", icon="fiche", pct=100)
    with c4:
        kpi_card("Élèves primaire", _edu_fmt("eleves_primaire", "int"),
                 f"inscrits · {_edu_fmt('eleves_primaire', 'annee')}", icon="pop", pct=100)
    with c5:
        kpi_card("Enseignants", _edu_fmt("enseignants_primaire", "int"),
                 f"primaire · {_edu_fmt('enseignants_primaire', 'annee')}", icon="pop", pct=100)

    # ── Bandeau 2 : TBS ──────────────────────────────────────────
    section("Taux bruts de scolarisation (TBS) par niveau", "analyse")
    tbs = [("tbs_prescolaire", "TBS Préscolaire", "warn", "fiche"),
           ("tbs_primaire", "TBS Primaire", "ok", "check"),
           ("tbs_college", "TBS Collège", "warn", "check"),
           ("tbs_lycee", "TBS Lycée", "danger", "alert")]
    cols = st.columns(4)
    for col, (key, lbl, var, icn) in zip(cols, tbs):
        with col:
            crit = var == "danger"
            kpi_card(lbl, _edu_fmt(key, "pct"),
                     f"réf. {_edu_fmt(key, 'annee')}" + (" — CRITIQUE" if crit else ""),
                     var, icon=icn, pct=_edu_v(key), pulse=crit)

    # 📘 Un TBS peut dépasser 100 % (élèves plus âgés ou plus jeunes que l'âge officiel).
    #    On utilise des barres (et non un "entonnoir") car les niveaux ne sont pas
    #    des étapes qui se vident : 37 % → 95 % → 77 % → 34 % n'est pas décroissant.
    fig_tbs = go.Figure(go.Bar(
        x=["Préscolaire", "Primaire", "Collège", "Lycée"],
        y=[_edu_v(k) for k, *_ in tbs],
        marker_color=["#d97706", "#059669", "#d97706", C_RED],
        text=[f"{_edu_v(k):.1f} %" for k, *_ in tbs], textposition="outside"))
    fig_tbs.update_layout(yaxis=dict(range=[0, 110]), showlegend=False)
    st.plotly_chart(_clean_fig(fig_tbs, 320, "DIES · opendata.gouv.tg",
                               title="Scolarisation par niveau (TBS, %)", y_title="TBS (%)"),
                    width="stretch")

    # ── Bandeau 3 : parité + encadrement ─────────────────────────
    section("Parité Filles/Garçons (IPS) & encadrement", "scale")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("IPS Primaire", _edu_fmt("ips_primaire", "ratio"),
                 f"F/G · {_edu_fmt('ips_primaire', 'annee')} — quasi-parité", "ok", icon="scale",
                 pct=100 * _edu_v("ips_primaire"))
    with c2:
        kpi_card("IPS Collège", _edu_fmt("ips_college", "ratio"),
                 f"F/G · {_edu_fmt('ips_college', 'annee')} — inégalité", "warn", icon="scale",
                 pct=100 * _edu_v("ips_college"))
    with c3:
        kpi_card("IPS Lycée", _edu_fmt("ips_lycee", "ratio"),
                 f"F/G · {_edu_fmt('ips_lycee', 'annee')} — CRITIQUE", "danger", icon="alert",
                 pct=100 * _edu_v("ips_lycee"), pulse=True)
    with c4:
        pe = _edu_v("ratio_pe_primaire", 40.7)
        kpi_card("Ratio P/E", f"{pe:.1f}", "élèves/ens. · norme UNESCO ≤ 40",
                 "danger" if pe > 40 else "ok", icon="alert", pct=min(pe / 40 * 100, 100))

    # ── Bandeau 4 : infrastructure ───────────────────────────────
    section("Infrastructure scolaire", "qualite")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Électricité (primaire)", _edu_fmt("electrification_primaire", "pct"),
                 f"réf. {_edu_fmt('electrification_primaire', 'annee')}", "warn", icon="scale",
                 pct=_edu_v("electrification_primaire"))
    with c2:
        kpi_card("Eau (primaire)", _edu_fmt("eau_primaire", "pct"),
                 f"réf. {_edu_fmt('eau_primaire', 'annee')}", "warn", icon="eau",
                 pct=_edu_v("eau_primaire"))
    with c3:
        kpi_card("Internet (primaire)", _edu_fmt("internet_primaire", "pct"),
                 f"réf. {_edu_fmt('internet_primaire', 'annee')} — CRITIQUE", "danger", icon="alert",
                 pct=_edu_v("internet_primaire"), pulse=True)
    with c4:
        kpi_card("Électricité (lycée)", _edu_fmt("electrification_lycee", "pct"),
                 f"réf. {_edu_fmt('electrification_lycee', 'annee')}", "ok", icon="check",
                 pct=_edu_v("electrification_lycee"))

    box("info",
        "<strong>Sources</strong> : DIES (statistiques scolaires, ≤ 2022) · RSA (Recensement scolaire "
        "annuel, 2024) · RGPH (population, 2022) · ODD 4 (2024), via <strong>opendata.gouv.tg</strong> "
        "et <strong>geodata.gouv.tg</strong>. L'année de référence est affichée sous chaque KPI.",
        "Sources officielles")

    tab1, tab2, tab3, tab4 = st.tabs(["Vue d'ensemble", "Défi 1 · Décrochage",
                                      "Défi 2 · Inégalités", "Défi 3 · Optimisation & simulation"])

    # ═══ TAB 1 — Vue d'ensemble ═══════════════════════════════════
    with tab1:
        section("Carte nationale des établissements", "map")
        if not communes_df.empty and "nb_ecoles" in communes_df.columns:
            try:
                show_map(folium_map(communes_df, "nb_ecoles", "Écoles par commune"),
                         460, key="edu_map_ecoles")
                st.caption("Densité d'établissements scolaires géolocalisés par commune · Source RSA 2024")
            except Exception as e:
                box("alert", f"Carte indisponible : {e}")
        else:
            box("info", "Données géolocalisées par école non chargées dans ce profil. La carte "
                "s'activera automatiquement si la colonne <code>nb_ecoles</code> est présente.")

        section("Synthèse nationale", "fiche")
        st.markdown(f"""
### État du système éducatif togolais (2019-2024)

Le Togo compte **{_edu_fmt('etablissements_geo', 'int')} établissements** géolocalisés (RSA 2024)
et **{_edu_fmt('batiments_scolaires', 'int')} bâtiments**, accueillant
**{_edu_fmt('eleves_primaire', 'int')} élèves** au primaire, encadrés par
**{_edu_fmt('enseignants_primaire', 'int')} enseignants** (DIES 2022).

**Points critiques**
- 🔴 **TBS lycée {_edu_fmt('tbs_lycee', 'pct')}** : moins d'un jeune sur 3 en âge d'aller au lycée y est scolarisé
- 🔴 **IPS lycée {_edu_fmt('ips_lycee', 'ratio')}** : net désavantage des filles
- 🔴 **Internet {_edu_fmt('internet_primaire', 'pct')}** : quasi-absence de connectivité dans les écoles primaires
- 🟠 **Ratio P/E {_edu_v('ratio_pe_primaire'):.1f}** : légèrement au-dessus de la norme UNESCO (40)

**Points positifs**
- ✅ **TBS primaire {_edu_fmt('tbs_primaire', 'pct')}** : scolarisation primaire quasi universelle
- ✅ **IPS primaire {_edu_fmt('ips_primaire', 'ratio')}** : parité quasi atteinte au primaire
- ✅ **Électrification lycée {_edu_fmt('electrification_lycee', 'pct')}** : second cycle bien équipé
""")

    # ═══ TAB 2 — Décrochage ═══════════════════════════════════════
    with tab2:
        section("Défi 1 — Anticiper le décrochage scolaire", "alert")
        has_regional = (not regions_df.empty and
                        any(c in regions_df.columns for c in ("abandon", "achevement", "redoublement")))

        if has_regional:
            reg = regions_df.copy()
            # 📘 Risque composite (0-100) = 50 % abandon + 30 % redoublement + 20 % (100 − achèvement),
            #    chaque variable étant d'abord ramenée à une échelle 0-100 entre sa plus petite
            #    et sa plus grande valeur régionale.
            reg["risque_decrochage"] = 0.0
            for col, poids in {"abandon": 0.5, "redoublement": 0.3, "achevement": -0.2}.items():
                s = _num_col(reg, col)
                lo, hi = s.min(), s.max()
                if pd.notna(lo) and pd.notna(hi) and hi > lo:
                    norm = (s - lo) / (hi - lo) * 100
                    if poids < 0:
                        norm = 100 - norm
                    reg["risque_decrochage"] += norm.fillna(0) * abs(poids)
            reg["risque_decrochage"] = reg["risque_decrochage"].round(0)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                v = _num_col(reg, "abandon").mean()
                v = 0 if pd.isna(v) else v
                kpi_card("Abandon moyen", f"{v:.1f} %", "primaire",
                         "danger" if v > 15 else "warn", pct=min(v, 100))
            with c2:
                v = _num_col(reg, "achevement").mean()
                v = 0 if pd.isna(v) else v
                kpi_card("Achèvement moyen", f"{v:.1f} %", "primaire",
                         "ok" if v > 80 else "warn", pct=min(v, 100))
            with c3:
                v = _num_col(reg, "redoublement").mean()
                v = 0 if pd.isna(v) else v
                kpi_card("Redoublement moyen", f"{v:.1f} %", "primaire", "warn", pct=min(v, 100))
            with c4:
                n_crit = int((reg["risque_decrochage"] >= 60).sum())
                kpi_card("Régions critiques", f"{n_crit}/{len(reg)}", "risque ≥ 60",
                         "danger" if n_crit > 0 else "ok", icon="alert")

            section("Carte du risque composite", "map")
            try:
                show_map(folium_map(reg, "risque_decrochage", "Risque de décrochage (0-100)"),
                         440, key="edu_map_decrochage")
            except Exception:
                pass

            section("Détail par région", "fiche")
            cols = [c for c in ("commune", "region", "abandon", "achevement",
                                "redoublement", "risque_decrochage") if c in reg.columns]
            rename = {"commune": "Territoire", "region": "Région", "abandon": "Abandon (%)",
                      "achevement": "Achèvement (%)", "redoublement": "Redoubl. (%)",
                      "risque_decrochage": "Risque (0-100)"}
            df = reg[cols].rename(columns=rename)
            if "Risque (0-100)" in df.columns:
                df = df.sort_values("Risque (0-100)", ascending=False)
            st.dataframe(df, width="stretch", hide_index=True)

            section("Recommandations par région", "alert")
            # Les encadrés utilisent les styles "alert / tip / ok" définis dans le CSS.
            for _, r in reg.sort_values("risque_decrochage", ascending=False).iterrows():
                score = float(r.get("risque_decrochage", 0))
                nom = _nom_territoire(r)
                if score >= 60:
                    niv, kind, act = "🔴 CRITIQUE", "alert", "Bourses ciblées + cantines scolaires"
                elif score >= 40:
                    niv, kind, act = "🟠 ÉLEVÉ", "tip", "Suivi des élèves + appui aux familles"
                elif score >= 20:
                    niv, kind, act = "🟡 MODÉRÉ", "tip", "Prévention + sensibilisation"
                else:
                    niv, kind, act = "🟢 FAIBLE", "ok", "Maintenir les acquis"
                box(kind, f"{niv} (score {score:.0f}/100) → <em>{act}</em>", nom)
        else:
            box("info", "Les taux d'abandon et de redoublement par région ne sont pas exposés sur les "
                "portails. Affichage de l'<strong>indice composite de risque</strong> "
                "(TBS + IPS + infrastructure).")
            c1, c2 = st.columns(2)
            with c1:
                top = EDU_RISQUE_REGIONS.sort_values("Risque décrochage (0-100)", ascending=False).iloc[0]
                kpi_card("Région la plus à risque", top["Région"],
                         f"indice {top['Risque décrochage (0-100)']:.1f} / 100", "danger",
                         icon="alert", pulse=True)
            with c2:
                kpi_card("Préfectures prioritaires", " · ".join(EDU_RISQUE_TOP_PREF),
                         "top 3 du risque de décrochage", "warn", icon="alert")

            dfr = EDU_RISQUE_REGIONS.sort_values("Risque décrochage (0-100)")
            fig_risk = px.bar(dfr, x="Risque décrochage (0-100)", y="Région", orientation="h",
                              color="Statut", text="Risque décrochage (0-100)",
                              color_discrete_map={"Vérifié": C_GREEN, "À confirmer": "#cbd5e1"})
            fig_risk.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_risk.update_layout(xaxis=dict(range=[0, 100]))
            st.plotly_chart(_clean_fig(fig_risk, 380, "calcul VIGIE·TOGO",
                                       title="Indice de risque de décrochage par région"),
                            width="stretch")
            if (dfr["Statut"] == "À confirmer").any():
                box("tip", "Les barres grises (" + ", ".join(dfr.loc[dfr["Statut"] == "À confirmer", "Région"]) +
                    ") sont des valeurs <strong>à confirmer</strong> avant toute publication.",
                    "Fiabilité")

    # ═══ TAB 3 — Inégalités ═══════════════════════════════════════
    with tab3:
        section("Défi 2 — Mesurer les inégalités d'accès", "analyse")
        box("info", "IPS = Indice de Parité · <strong>1,0 = parité parfaite</strong> · "
            "&lt; 0,95 = inégalité · &lt; 0,80 = critique")

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("IPS Primaire", _edu_fmt("ips_primaire", "ratio"),
                     f"F/G · {_edu_fmt('ips_primaire', 'annee')}", "ok", icon="scale",
                     pct=100 * _edu_v("ips_primaire"))
        with c2:
            kpi_card("IPS Collège", _edu_fmt("ips_college", "ratio"),
                     f"F/G · {_edu_fmt('ips_college', 'annee')}", "warn", icon="scale",
                     pct=100 * _edu_v("ips_college"))
        with c3:
            kpi_card("IPS Lycée", _edu_fmt("ips_lycee", "ratio"),
                     f"F/G · {_edu_fmt('ips_lycee', 'annee')} — CRITIQUE", "danger", icon="alert",
                     pct=100 * _edu_v("ips_lycee"), pulse=True)

        ips_vals = [_edu_v("ips_primaire"), _edu_v("ips_college"), _edu_v("ips_lycee")]
        fig_ips = go.Figure(go.Bar(
            x=["Primaire", "Collège", "Lycée"], y=ips_vals,
            marker_color=["#059669", "#d97706", C_RED],
            text=[f"{v:.2f}" for v in ips_vals], textposition="outside"))
        fig_ips.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Parité")
        fig_ips.add_hline(y=0.95, line_dash="dot", line_color="orange", annotation_text="Seuil 0,95")
        fig_ips.update_layout(yaxis=dict(range=[0, 1.15]), showlegend=False)
        st.plotly_chart(_clean_fig(fig_ips, 380, "DIES · opendata.gouv.tg",
                                   title="Indice de parité par niveau", y_title="IPS (F/G)"),
                        width="stretch")

        filles = round(100 * _edu_v("ips_lycee"))
        box("alert", f"IPS = {_edu_v('ips_lycee'):.2f} : pour 100 garçons scolarisés au lycée, seulement "
            f"{filles} filles le sont. Actions prioritaires : bourses pour filles, internats sécurisés, "
            "sensibilisation des familles.", "Désavantage critique au lycée")

        section("Points chauds — IPS lycée par préfecture", "map")
        fig_hot = px.bar(EDU_IPS_HOTSPOTS.sort_values("IPS lycée (F/G)"),
                         x="IPS lycée (F/G)", y="Préfecture", orientation="h",
                         color="IPS lycée (F/G)", color_continuous_scale="RdYlGn",
                         range_color=[0.2, 1.0], text="IPS lycée (F/G)")
        fig_hot.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_hot.add_vline(x=1.0, line_dash="dash", line_color="green", annotation_text="Parité")
        fig_hot.update_layout(coloraxis_showscale=False, xaxis=dict(range=[0, 1.1]))
        st.plotly_chart(_clean_fig(fig_hot, 300, "MEN · opendata.gouv.tg",
                                   title="Les 3 préfectures les plus inégalitaires au lycée"),
                        width="stretch")

        section("Inégalités urbain / rural", "scale")
        box("info", "Le croisement fin du recensement RSA 2024 avec la classification urbain/rural "
            "n'est pas exposé sur les portails. Il est prévu en mise à jour.", "Données à venir")

    # ═══ TAB 4 — Optimisation & simulation ════════════════════════
    with tab4:
        section("Défi 3 — Optimiser la répartition des écoles & enseignants", "projets")

        has_schools = (not communes_df.empty and
                       any(_num_col(communes_df, c).notna().any()
                           for c in ("nb_maternelle", "nb_primaire", "nb_college", "nb_lycee")))
        if has_schools:
            c1, c2, c3, c4 = st.columns(4)
            for col, (cn, lbl, sub) in zip((c1, c2, c3, c4), [
                    ("nb_maternelle", "Maternelles", "préscolaires"),
                    ("nb_primaire", "Primaires", "écoles"),
                    ("nb_college", "Collèges", "1er cycle"),
                    ("nb_lycee", "Lycées", "2nd cycle")]):
                with col:
                    n = int(_num_col(communes_df, cn).fillna(0).sum())
                    kpi_card(lbl, f"{n:,}".replace(",", " "), sub, icon="fiche")
        else:
            box("info", "Répartition détaillée par niveau non chargée dans ce profil.")

        section("Simulation — Redéploiement des enseignants du primaire", "scenario")
        st.markdown(
            "**Objectif** : rapprocher le ratio élèves/enseignant de la norme UNESCO en redéployant "
            "des enseignants depuis les inspections sur-dotées (Lomé-Aéroport, Lomé-Centre, "
            "Sotouboua-Nord…) vers les sous-dotées.")

        cible = st.slider("Cible P/E (élèves par enseignant)", 35, 50, 40, 1, key="edu_sim_cible")
        ens_redep, eleves_gagnes = _edu_interp_sim(cible)
        exact = cible in {a[0] for a in EDU_SIM_ANCHORS}
        pe = _edu_v("ratio_pe_primaire", 40.7)
        max_ens = max(a[1] for a in EDU_SIM_ANCHORS)
        max_el = max(a[2] for a in EDU_SIM_ANCHORS)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Enseignants à redéployer", f"{ens_redep:,}".replace(",", " "),
                     f"cible {cible} él/ens · " + ("valeur vérifiée" if exact else "estimation"),
                     "warn" if ens_redep > 500 else "ok", icon="pop",
                     pct=min(100 * ens_redep / max_ens, 100))
        with c2:
            kpi_card("Élèves gagnants", f"{eleves_gagnes:,}".replace(",", " "),
                     "passent sous la cible · " + ("valeur vérifiée" if exact else "estimation"),
                     "ok", icon="check", pct=min(100 * eleves_gagnes / max_el, 100))
        with c3:
            kpi_card("Ratio P/E actuel", f"{pe:.1f}", f"cible {cible} → écart {pe - cible:+.1f}",
                     "danger" if pe > cible else "ok", icon="scale")

        # Courbe : les 3 points vérifiés + le point choisi.
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=[a[0] for a in EDU_SIM_ANCHORS], y=[a[1] for a in EDU_SIM_ANCHORS],
                                     mode="lines+markers", name="Ancres vérifiées",
                                     marker_color=C_GREEN))
        fig_sim.add_trace(go.Scatter(x=[cible], y=[ens_redep], mode="markers",
                                     name=f"Cible choisie ({cible})",
                                     marker=dict(color=C_RED, size=14, symbol="diamond")))
        fig_sim.update_layout(hovermode="x unified")
        st.plotly_chart(_clean_fig(fig_sim, 360, "simulation (3 ancres vérifiées)",
                                   title="Redéploiement selon la cible",
                                   x_title="Cible P/E", y_title="Enseignants à redéployer"),
                        width="stretch")
        st.caption("Seules les cibles 35, 40 et 50 sont des résultats vérifiés ; les autres valeurs "
                   "sont interpolées linéairement entre ces trois points.")

        section("Origine des enseignants redéployables (3 premières inspections)", "map")
        st.dataframe(EDU_SIM_ORIGINE, width="stretch", hide_index=True)

        fmt = lambda n: f"{n:,}".replace(",", " ")  # noqa: E731
        if cible <= 40:
            box("ok", f"Avec une cible de {cible} élèves/enseignant, {fmt(ens_redep)} enseignants "
                f"redéployés font gagner {fmt(eleves_gagnes)} élèves.", "Objectif atteignable")
        else:
            box("tip", f"Une cible de {cible} est moins exigeante que la norme UNESCO (40). "
                "Pour un impact maximal, visez 40 ou moins.")


# ─────────────────────── CATALOGUE DES PROJETS ───────────────────────
def page_catalogue_projets():
    hero("Catalogue de tous les projets", "Tous secteurs COSO.", chips=["Catalogue"])
    catalog = levels.get("_projects_catalog", pd.DataFrame())
    if catalog.empty:
        box("alert", "Aucun projet chargé.")
        return

    n_total = len(catalog)
    n_ach = int(catalog["statut"].astype(str).str.contains("Achevé|Réceptionné", na=False).sum())
    cout_total = pd.to_numeric(catalog["cout_total"], errors="coerce").fillna(0).sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Total projets", n_total, f"{catalog['secteur'].nunique()} secteurs", icon="projets")
    with c2:
        kpi_card("Achevés", n_ach, f"{100 * n_ach / max(n_total, 1):.0f} %", "ok", icon="check")
    with c3:
        kpi_card("Budget total", f"{cout_total / 1e6:.0f} M FCFA" if cout_total > 0 else "N/D",
                 icon="scale")

    # Filtres rapides sur le catalogue
    f1, f2 = st.columns(2)
    sect = f1.multiselect("Secteur", sorted(catalog["secteur"].dropna().unique()))
    stat = f2.multiselect("Statut", sorted(catalog["statut"].dropna().unique()))
    df = catalog.copy()
    if sect:
        df = df[df["secteur"].isin(sect)]
    if stat:
        df = df[df["statut"].isin(stat)]

    cols = [c for c in ["project_id", "titre", "secteur", "statut", "commune", "cout_total"]
            if c in df.columns]
    df_show = df[cols].copy()
    if "cout_total" in df_show.columns:
        df_show["cout_total"] = (pd.to_numeric(df_show["cout_total"], errors="coerce") / 1e6).round(1)
    df_show = df_show.rename(columns={"project_id": "ID", "titre": "Titre", "secteur": "Secteur",
                                      "statut": "Statut", "commune": "Commune",
                                      "cout_total": "Coût (M FCFA)"})
    st.caption(f"{len(df_show)} projet(s) affiché(s)")
    st.dataframe(df_show, width="stretch", hide_index=True, height=480)


# ─────────────────────── DIAGNOSTIC NORMATIF ───────────────────────
def page_normes():
    hero("Diagnostic normatif", "Écarts par rapport aux standards OMS, Sphère, IPC.",
         chips=["OMS", "Sphère", "IPC"])
    if gdf_view.empty:
        return

    tot = len(gdf_view)
    section("Territoires sous les normes — cliquez pour les lister", "alert")
    kpis = [("alerte_sante", "Santé", "OMS", "sante"),
            ("alerte_eau", "Eau", "Sphère", "eau"),
            ("alerte_alim", "Alimentation", "norme", "alimentation"),
            ("alerte_ino", "Inondation", "FRI > 60", "inondation")]
    cols = st.columns(4)
    for i, (col, lbl, sub, icn) in enumerate(kpis):
        n = int(gdf_view[col].fillna(False).sum()) if col in gdf_view.columns else 0
        with cols[i]:
            # Au clic : tableau de bord filtré sur cette alerte.
            kpi_card(lbl, f"{n}/{tot}", sub, "danger" if n > 0 else "ok", icon=icn,
                     pct=100 * n / max(tot, 1), target="dashboard",
                     state={"focus": col}, hint="Lister ces territoires")

    section("Tableau détaillé", "fiche")
    cd = {"commune": "Territoire", "region": "Région",
          "health_fac_per_10000": "Formations/10k", "alerte_sante": "Santé",
          "water_pts_per1000": "Eau/1000", "alerte_eau": "Eau",
          "food_markets_per_10000": "Marchés/10k", "alerte_alim": "Alim",
          "flood_fsi_max": "FRI max", "alerte_ino": "Inond"}
    cp = [c for c in cd if c in gdf_view.columns]
    dfd = gdf_view[cp].rename(columns=cd).copy()
    bc = [c for c in dfd.columns if c in ("Santé", "Eau", "Alim", "Inond")]
    st.dataframe(style_alerts(dfd, bc), width="stretch", height=440)


# ─────────────────────── FICHE TERRITORIALE ───────────────────────
def page_fiche():
    hero("Fiche territoriale", "Profil détaillé et recommandations.",
         chips=["Radar 4 piliers", "Recommandation"])
    if gdf.empty:
        return

    bouton_imprimer("Imprimer cette fiche")

    noms = sorted(gdf["commune"].dropna().unique())
    tgt = st.session_state.pop("fiche_target", None)    # posé par un bouton « Ouvrir la fiche »
    if tgt in noms:
        st.session_state["fiche_select"] = tgt
    choix = st.selectbox("Sélectionner un territoire", noms, key="fiche_select")
    row = gdf[gdf["commune"] == choix].iloc[0]

    pop = (f"{int(row['population']):,}".replace(",", " ")
           if pd.notna(row.get("population")) else "N/D")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Région", str(row.get("region", "N/D")), icon="map")
    with c2:
        kpi_card("Population", pop, "habitants", icon="pop")
    with c3:
        kpi_card("ICV", f"{row['ICV']:.0f}" if pd.notna(row.get("ICV")) else "N/D",
                 str(row.get("classe", "")) if pd.notna(row.get("classe")) else "",
                 "danger" if row.get("classe") == "Critique" else "warn", icon="scale",
                 pct=row.get("ICV", 0))
    with c4:
        dp = bool(row.get("double_peine"))
        kpi_card("Double peine", "Oui" if dp else "Non", "services + inondation",
                 "danger" if dp else "ok", icon="alert",
                 target="analyse" if dp else None, hint="Voir la matrice")

    section("Profil de vulnérabilité (comparé à la moyenne)", "analyse")
    cats = [PILLAR_LABEL[p] for p in PILLARS]
    def _val(r, p):
        k = f"score_{p}"
        return float(r[k]) if k in r.index and pd.notna(r[k]) else 0.0
    vals = [_val(row, p) for p in PILLARS]
    moy = [float(gdf[f"score_{p}"].mean()) if f"score_{p}" in gdf.columns else 0.0 for p in PILLARS]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=moy + [moy[0]], theta=cats + [cats[0]], name="Moyenne du niveau",
                                    line=dict(color="#94a3b8", dash="dash")))
    radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], name=choix,
                                    fill="toself", line_color=C_GREEN,
                                    fillcolor="rgba(0,106,78,.25)"))
    radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(_clean_fig(radar, 440, "calcul VIGIE·TOGO", title=f"Radar — {choix}"),
                    width="stretch")

    worst = max(PILLARS, key=lambda p: _val(row, p))
    reco = {"sante": "Prioriser une formation sanitaire.",
            "eau": "Prioriser forages et points d'eau.",
            "alimentation": "Stock tampon et stabilisation des prix.",
            "inondation": "Intégrer l'aléa inondation au plan d'aménagement.",
            "erosion": "Renforcer les ouvrages de protection côtière."}[worst]
    box("alert" if row.get("classe") == "Critique" else "tip",
        f"<strong>Pilier critique : {PILLAR_LABEL[worst]}</strong> → {reco}", title="Recommandation")

    b1, b2, _sp = st.columns([1, 1.4, 2])
    with b1:
        if st.button("← Tableau de bord", key="fiche_back"):
            st.switch_page(NAV_PAGES["dashboard"])
    with b2:
        if st.button("🗳 Préparer une décision →", key="fiche_decision", type="primary"):
            st.session_state["decision_target"] = choix
            st.switch_page(NAV_PAGES["decision"])

    if mode_citoyen:
        section("Fiche citoyenne — imprimable", "fiche")
        st.markdown(f"""
        <div class="fiche-citoyen">
          <h2>{choix}</h2>
          <p style="color:#6c757d; font-size:.85rem">Région {row.get('region', 'N/D')} · Population : {pop}</p>
          <div class="fiche-big">{row['ICV']:.0f}<small>Indice de vulnérabilité (0–100)</small></div>
          <div style="text-align:center; margin:.8rem 0">{badge_classe(row.get('classe', ''))}</div>
          <div class="fiche-reco"><strong>Action prioritaire :</strong> {reco}</div>
        </div>""", unsafe_allow_html=True)
        bouton_imprimer("Imprimer la fiche citoyenne")


# ─────────────────────── PRIX & PRÉVISIONS ───────────────────────
def page_prix():
    hero("Prix & prévisions", "Prix PAM (WFP) — données 2001-2022.", chips=["PAM / WFP"])
    if prices.empty:
        box("alert", "Aucune donnée de prix.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        d = st.selectbox("Denrée", sorted(prices["denree"].unique()))
    with c2:
        marches = sorted(prices["marche"].unique())
        sel = st.multiselect("Marchés", marches, default=marches[:4])

    sub = prices[(prices["denree"] == d) & (prices["marche"].isin(sel))]
    if sub.empty:
        box("tip", "Sélectionnez au moins un marché.")
        return

    fig = px.line(sub, x="date", y="prix_fcfa_kg", color="marche", markers=True,
                  labels={"prix_fcfa_kg": "Prix (FCFA/kg)", "date": "", "marche": "Marché"})
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(_clean_fig(fig, 450, "PAM/WFP", title=f"Prix : {d}"), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Relevés", len(sub), "sur sélection", icon="sources")
    with c2:
        kpi_card("Prix moyen", f"{sub['prix_fcfa_kg'].mean():.0f} FCFA", "par kg", icon="prix")
    with c3:
        kpi_card("Marchés", sub["marche"].nunique(), "suivis", icon="map")


# ─────────────────────── QUALITÉ DES DONNÉES ───────────────────────
def page_qualite():
    hero("Qualité des données", "Couverture, complétude, audit.", chips=["Audit"])
    section("Volumes et complétude par niveau", "qualite")
    rows = []
    for name, g in levels.items():
        if name.startswith("_") or not isinstance(g, pd.DataFrame) or g.empty:
            continue
        data = g.drop(columns="geometry", errors="ignore")
        rows.append({"Niveau": name.capitalize(), "Territoires": len(g),
                     "Colonnes": data.shape[1],
                     "Complétude (%)": round(100 * (1 - data.isna().mean().mean()), 1)})
    dfq = pd.DataFrame(rows)
    st.dataframe(dfq, width="stretch", hide_index=True,
                 column_config={"Complétude (%)": st.column_config.ProgressColumn(
                     "Complétude (%)", min_value=0, max_value=100, format="%.1f %%")})

    section("Incohérences", "alert")
    alertes = _check_pciaeph_coherence()
    if alertes:
        for a in alertes:
            box("alert", a)
    else:
        box("ok", "Aucune incohérence détectée.")


# ─────────────────────── SOURCES & LICENCES ───────────────────────
def page_sources():
    hero("Sources & licences", "Traçabilité complète.", chips=["Open data"])
    reg = get_sources_registry()
    if not reg:
        box("alert", "Aucune source enregistrée.")
        return

    n_ok = sum(1 for v in reg.values() if v.get("licence_ok"))
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Sources", len(reg), icon="sources")
    with c2:
        kpi_card("Licences OK", n_ok, "réutilisables", "ok", icon="check")
    with c3:
        kpi_card("À clarifier", len(reg) - n_ok, "avant publication", "warn", icon="alert")

    rows = [{"Source": v["titre"], "Producteur": v.get("producteur", "N/D"),
             "Licence": v["licence"],
             "Réutilisable": "Oui" if v.get("licence_ok") else "À clarifier",
             "Mise à jour": v.get("date_maj", "N/D")} for v in reg.values()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=400)


# ═════════════════════════════════════════════════════════════════════════
# AIDE CONTEXTUELLE, MÉTHODOLOGIE COMPLÈTE ET AIDE À LA DÉCISION
# ═════════════════════════════════════════════════════════════════════════
# 📘 POUR LES DÉBUTANTS : ce bloc contient trois choses liées.
#   1. PAGES_HELP       : le texte d'aide de CHAQUE page (une seule source de vérité).
#                         Il s'affiche (a) dans le bandeau « Comment lire cette page ? »
#                         en haut de chaque page, (b) dans la page « Méthodologie & guide ».
#   2. GLOSSAIRE        : tous les sigles et termes de l'application.
#   3. page_decision()  : l'aide à la décision. L'application PROPOSE, l'utilisateur
#                         VALIDE, MODIFIE ou REJETTE, et chaque décision est tracée.
import json as _json
from datetime import datetime as _dt

# ────────────────────────────────────────────────────────────────
# 1. TEXTES D'AIDE PAR PAGE
# ────────────────────────────────────────────────────────────────
PAGES_HELP = {
    "dashboard": {
        "objectif": "Voir d'un coup d'œil quels territoires sont les plus vulnérables, sur une carte, un classement, un histogramme ou un tableau.",
        "lire": [
            "**Carte** : plus la couleur est foncée (rouge), plus le territoire est vulnérable. Gris = donnée absente.",
            "**Indicateur affiché** : choisissez l'ICV global ou un seul pilier (Santé, Eau…). Tout l'écran suit votre choix.",
            "**KPI du haut** : Territoires (nombre affiché), Critiques (ICV > 75), Double peine (manque de services ET risque d'inondation), ICV moyen.",
            "**Top 15** : les 15 territoires les plus vulnérables pour l'indicateur choisi.",
            "**Distribution** : combien de territoires se trouvent dans chaque tranche de 0 à 100.",
        ],
        "cliquer": "Cliquez un KPI pour filtrer. Cliquez un territoire (carte, barre, ligne du tableau) puis « Ouvrir la fiche ». Le bouton « ✕ Retirer » enlève le filtre actif.",
        "attention": "L'ICV est un indicateur RELATIF : il compare les territoires entre eux. Il ne mesure pas un niveau de vie absolu.",
    },
    "analyse": {
        "objectif": "Repérer les territoires à traiter en urgence en croisant deux problèmes : le manque de services et le risque d'inondation.",
        "lire": [
            "**Matrice** : chaque bulle est un territoire ; sa taille = sa population ; sa couleur = son ICV.",
            "Les deux lignes pointillées découpent 4 cadrans : **Stable** (en bas à gauche), **À équiper** (déficit de services élevé), **À protéger** (exposition inondation élevée), **URGENCE** (les deux).",
            "**Hot-spots** : territoires qui cumulent 3 alertes ou plus (Santé, Eau, Alimentation, Inondation, Érosion).",
        ],
        "cliquer": "Cliquez une bulle ou une ligne du tableau des hot-spots, puis « Ouvrir la fiche ».",
        "attention": "Les seuils des cadrans (60 en abscisse, 55 en ordonnée) sont des conventions de lecture, pas des normes officielles.",
    },
    "scenarios": {
        "objectif": "Tester « et si on améliorait tel pilier de X % ? » et voir l'effet sur l'ICV et sur le nombre de territoires critiques.",
        "lire": [
            "**Piliers à améliorer** : ceux dont on réduit le déficit.",
            "**Réduction (%)** : de combien on diminue le score de ces piliers (hypothèse de départ).",
            "**Critiques A → B** : nombre de territoires critiques avant → après. **ICV moyen** idem.",
            "**Histogramme avant/après** : le gris est la situation actuelle, le vert la situation simulée.",
        ],
        "cliquer": "Le KPI « Critiques » ouvre le tableau de bord filtré sur les critiques actuels.",
        "attention": "Une simulation est une hypothèse de travail, pas une prévision : elle ne dit pas combien coûterait l'amélioration ni si elle est réalisable.",
    },
    "decision": {
        "objectif": "Transformer l'analyse en décisions concrètes, traçables et validées par une personne.",
        "lire": [
            "**Propositions** : générées par des règles simples (piliers au-dessus de 50/100), classées Urgent (≥ 75), Prioritaire (≥ 50) ou Veille.",
            "**« Pourquoi ? »** : sous chaque proposition, les chiffres qui la justifient (score, valeur observée, norme).",
            "**Votre décision** : Validée, Validée avec modification, À étudier plus tard ou Rejetée.",
        ],
        "cliquer": "Choisissez un territoire (liste de priorité ou menu), décidez pour chaque proposition, donnez votre note d'utilité puis enregistrez.",
        "attention": "Rien n'est appliqué automatiquement. Votre nom, la date et votre justification sont enregistrés dans le journal des décisions.",
    },
    "education": {
        "objectif": "Suivre l'éducation sous trois angles : décrochage, inégalités filles/garçons, répartition des écoles et des enseignants.",
        "lire": [
            "**TBS** (Taux Brut de Scolarisation) : élèves inscrits dans un cycle ÷ population en âge de le suivre. Peut dépasser 100 %.",
            "**IPS** (Indice de Parité entre les Sexes) : filles ÷ garçons. 1,00 = parité ; sous 0,95 = inégalité ; sous 0,80 = critique.",
            "**Ratio P/E** : nombre d'élèves par enseignant. La norme de référence est 40 au maximum.",
            "**Simulation** : combien d'enseignants redéployer pour atteindre un ratio cible.",
        ],
        "cliquer": "Utilisez le curseur « Cible P/E » de l'onglet Optimisation. Les années de référence sont indiquées sous chaque KPI.",
        "attention": "Les chiffres nationaux datent de 2019 à 2024 selon la source. Les valeurs marquées « À confirmer » ne doivent pas être publiées telles quelles.",
    },
    #"denrees": {
    #    "objectif": "Suivre les denrées alimentaires et l'accès aux marchés par territoire.",
    #    "lire": ["Les KPI résument les prix et la couverture en marchés ; les cartes colorent les territoires selon l'indicateur choisi."],
    #    "cliquer": "Utilisez les filtres de la page pour changer de denrée ou de marché.",
    #    "attention": "Les prix proviennent du PAM (Programme alimentaire mondial) et s'arrêtent en 2022 : voir « Sources & licences ».",
    #},
    "projets": {
        "objectif": "Suivre les chantiers (projets hydrauliques) et les zones agricoles aménagées (ZAAP).",
        "lire": [
            "**Projets** : nombre total ; **Achevés** : terminés ou réceptionnés ; **En cours** : chantiers actifs ; **Budget** : coût cumulé en millions de FCFA.",
            "**ZAAP** : nombre de zones agricoles aménagées et surface protégée en hectares (ha).",
        ],
        "cliquer": "Chaque KPI de projets ouvre le catalogue détaillé.",
        "attention": "Cette page n'existe qu'au niveau Communes. Un « N/D » signifie donnée non disponible, pas zéro.",
    },
    "catalogue": {
        "objectif": "Consulter la liste complète des projets, tous secteurs confondus.",
        "lire": ["Chaque ligne est un projet : identifiant, titre, secteur, statut, commune et coût en millions de FCFA."],
        "cliquer": "Filtrez par secteur et par statut avec les deux menus au-dessus du tableau.",
        "attention": "Le coût affiché est en millions de FCFA ; un coût vide signifie qu'il n'est pas renseigné.",
    },
    "normes": {
        "objectif": "Comparer chaque territoire à des normes internationales (OMS, Sphère, etc.).",
        "lire": [
            "**KPI** : « 12/117 » = 12 territoires sous la norme sur 117.",
            "**Tableau** : les cellules rouges signalent un territoire sous la norme (alerte).",
        ],
        "cliquer": "Cliquez un KPI pour ouvrir le tableau de bord filtré sur cette alerte.",
        "attention": "La norme « Alimentation » (≥ 1 marché pour 10 000 habitants) est une approximation de travail, pas une norme officielle.",
    },
    "fiche": {
        "objectif": "Obtenir le profil complet d'un territoire et une recommandation principale, imprimable.",
        "lire": [
            "**Radar** : chaque branche est un pilier. Plus la zone verte s'étend, plus le territoire est vulnérable. La ligne grise pointillée = moyenne du niveau.",
            "**Pilier critique** : celui qui a le score le plus élevé ; la recommandation en découle.",
            "**Mode citoyen** (barre latérale) : affiche une fiche simplifiée à imprimer.",
        ],
        "cliquer": "« Préparer une décision » ouvre l'aide à la décision pour ce territoire. « Imprimer » ouvre l'impression du navigateur.",
        "attention": "La recommandation de la fiche est volontairement courte ; l'aide à la décision détaille les actions et garde la trace de votre choix.",
    },
    "prix": {
        "objectif": "Suivre l'évolution des prix des denrées sur plusieurs marchés.",
        "lire": ["Chaque courbe est un marché. Le survol affiche le prix en FCFA par kg. Les KPI donnent le nombre de relevés, le prix moyen et le nombre de marchés."],
        "cliquer": "Choisissez la denrée et les marchés à comparer.",
        "attention": "Données PAM (WFP) de 2001 à 2022 : elles ne reflètent pas les prix actuels.",
    },
    "qualite": {
        "objectif": "Savoir si les données sont complètes et cohérentes avant de s'y fier.",
        "lire": ["**Complétude** : part des cases renseignées par niveau. Plus elle est proche de 100 %, plus les résultats sont fiables.", "**Incohérences** : écarts détectés entre colonnes (ex. projets achevés > total)."],
        "cliquer": "Aucune action : page de contrôle.",
        "attention": "Une complétude faible rend les moyennes et les classements moins fiables.",
    },
    "sources": {
        "objectif": "Savoir d'où viennent les données et si elles sont réutilisables.",
        "lire": ["Chaque ligne : source, producteur, licence, réutilisable (oui / à clarifier) et date de mise à jour."],
        "cliquer": "Aucune action.",
        "attention": "Les licences « À clarifier » doivent être vérifiées avant toute publication.",
    },
}


def _render_help(h: dict) -> None:
    """Affiche un bloc d'aide (objectif / comment lire / comment agir / précautions)."""
    st.markdown(f"**À quoi sert cette page ?** {h['objectif']}")
    st.markdown("**Comment la lire**\n\n" + "\n".join(f"- {x}" for x in h["lire"]))
    st.markdown(f"**Comment agir** — {h['cliquer']}")
    st.markdown(f"**⚠ À garder en tête** — {h['attention']}")


def _contextual_help() -> None:
    """Bandeau pliable « Comment lire cette page ? », appelé automatiquement par hero()."""
    key = CURRENT.get("key")
    h = PAGES_HELP.get(key)
    if not h:
        return
    with st.expander("ℹ️ Comment lire cette page ?"):
        _render_help(h)
        if st.button("📖 Ouvrir le glossaire des sigles", key=f"help_gloss_{key}"):
            st.switch_page(NAV_PAGES["methode"])


# ────────────────────────────────────────────────────────────────
# 2. GLOSSAIRE (tous les sigles et termes de l'application)
# ────────────────────────────────────────────────────────────────
# (catégorie, sigle, signification, explication, signification_confirmée)
_A_CONF = ("Sigle repris des fichiers sources. Sa signification exacte est à confirmer "
           "auprès du producteur des données (voir « Sources & licences »).")
GLOSSAIRE = [
    # --- Lire l'application ---
    ("Application", "ICV", "Indice Communal de Vulnérabilité",
     "Note de 0 à 100 : plus elle est haute, plus le territoire est vulnérable. Combine les 4 piliers ICV actifs.", True),
    ("Application", "KPI", "Key Performance Indicator (indicateur clé)",
     "Chiffre mis en avant dans une carte colorée. Beaucoup de KPI sont cliquables.", True),
    ("Application", "Pilier", "Domaine de vulnérabilité",
     "Santé, Eau, Alimentation et Inondation sont les quatre piliers ICV actifs. Érosion conservée séparément.", True),
    ("Application", "Score de pilier", "Note 0-100 d'un pilier",
     "100 = situation la plus défavorable. Les 4 scores actifs alimentent l'ICV.", True),
    ("Application", "Pondération / poids", "Importance donnée à chaque pilier",
     "Réglable dans la barre latérale ; les poids sont ramenés à une somme de 1.", True),
    ("Application", "Classe", "Faible · Modérée · Élevée · Critique",
     "Faible < 25 · Modérée 25-50 · Élevée 50-75 · Critique > 75 (selon l'ICV).", True),
    ("Application", "Alerte", "Territoire sous une norme",
     "Drapeau rouge quand la valeur observée est moins bonne que la norme (Santé, Eau, Alimentation, Inondation, Érosion).", True),
    ("Application", "Double peine", "Manque de services + risque d'inondation",
     "Territoire cumulant un déficit de services de base et une forte exposition aux inondations.", True),
    ("Application", "Hot-spot", "Point chaud",
     "Territoire cumulant 3 alertes ou plus.", True),
    ("Application", "Maille territoriale", "Échelle d'analyse",
     "Régions, préfectures, communes ou cantons (choix dans la barre latérale).", True),
    ("Application", "Filtre actif / focus", "Filtre posé par un clic sur un KPI",
     "Affiché en haut du tableau de bord ; le bouton « ✕ Retirer » l'enlève.", True),
    ("Application", "Mode citoyen", "Affichage simplifié",
     "Langage simple et fiche imprimable dans « Fiche territoriale ».", True),
    ("Application", "Mode démonstration", "Données fictives",
     "Si la bannière rouge apparaît dans la barre latérale, les chiffres sont des exemples, pas des données réelles.", True),
    ("Application", "Journal des décisions", "Registre des choix validés",
     "Fichier CSV qui garde qui a décidé quoi, quand et pourquoi (page « Aide à la décision »).", True),
    # --- Santé ---
    ("Santé", "OMS", "Organisation mondiale de la Santé",
     "Référence pour la norme santé : au moins 2 formations sanitaires pour 10 000 habitants.", True),
    ("Santé", "Formation sanitaire", "Centre de santé, dispensaire, hôpital…",
     "Établissement où la population reçoit des soins.", True),
    # --- Eau ---
    ("Eau", "Sphère", "Standards humanitaires du Projet Sphère",
     "Référence pour la norme eau : au moins 4 points d'eau pour 1 000 habitants.", True),
    ("Eau", "Point d'eau", "Ouvrage d'approvisionnement en eau",
     "Forage, puits, borne ou équivalent, tel que recensé dans les données sources.", True),
    # --- Alimentation ---
    ("Alimentation", "PAM / WFP", "Programme alimentaire mondial / World Food Programme",
     "Agence de l'ONU qui publie les prix des denrées (2001-2022 ici).", True),
    ("Alimentation", "IPC", "Integrated Food Security Phase Classification",
     "Cadre intégré de classification de la sécurité alimentaire, en phases de 1 (minimale) à 5.", True),
    ("Alimentation", "FCFA", "Franc de la Communauté Financière Africaine",
     "Monnaie utilisée pour les prix et les budgets. « M FCFA » = millions de FCFA.", True),
    ("Alimentation", "Denrée", "Produit alimentaire de base",
     "Maïs, riz, sorgho, etc., suivi sur les marchés.", True),
    # --- Inondation / Érosion ---
    ("Inondation et érosion", "FRI", "Flood Risk Index (indice de risque d'inondation)",
     "Plus il est élevé, plus le risque est fort. Alerte au-dessus de 60.", True),
    ("Inondation et érosion", "FSI", "Flood Susceptibility Index (Indice de Susceptibilité aux Inondations)",
     "« FRI max / FSI max » = valeur la plus forte observée dans le territoire. ", True),
    ("Inondation et érosion", "EPR", "End Point Rate",
     "Vitesse d'évolution du trait de côte entre deux dates (mètres par an) : un recul signale de l'érosion.", True),
    ("Inondation et érosion", "Trait de côte", "Limite entre la terre et la mer",
     "Sa position est suivie pour mesurer l'érosion côtière.", True),
    # --- Projets ---
    ("Projets", "COSO", "Projet de Cohésion Sociale des régions Nord du Golfe de Guinée",
     "Il s'agit d'un projet financé par la Banque mondiale, couvrant 41 cantons frontaliers au Togo (dont 37 dans les Savanes), avec des investissements dans les infrastructures locales, les petits commerces et les solutions communautaires. Ce n'est donc pas un acronyme générique mais le nom d'un programme spécifique. ", True),
    ("Projets", "PCIAEPH", "Projet COSO - Infrastructures d'Alimentation en Eau Potable et Hydraulique au Togo",
     "Il présente les investissements hydrauliques du projet COSO dans les communautés rurales du Nord-Togo, avec 218 microprojets sur 12 cantons.Colonnes pciaeph_* : total, achevés, en cours, coût. ", True),
    ("Projets", "ZAAP", "Zones d'Aménagement Agricole Planifiées",
     "Zones agricoles aménagées ; on affiche leur nombre et leur surface en hectares.", True),
    ("Projets", "ha", "Hectare", "10 000 m².", True),
    # --- Éducation ---
    ("Éducation", "TBS", "Taux Brut de Scolarisation",
     "Élèves inscrits dans un cycle ÷ population en âge de le suivre. Peut dépasser 100 %.", True),
    ("Éducation", "IPS", "Indice de Parité entre les Sexes",
     "Filles ÷ garçons. 1,00 = parité ; < 0,95 inégalité ; < 0,80 critique.", True),
    ("Éducation", "P/E", "Ratio élèves par enseignant",
     "Nombre moyen d'élèves pour un enseignant. Norme de référence : 40 au maximum.", True),
    ("Éducation", "UNESCO", "Organisation des Nations unies pour l'éducation, la science et la culture",
     "Source de la norme de 40 élèves par enseignant.", True),
    ("Éducation", "ODD 4", "Objectif de Développement Durable n° 4",
     "Éducation de qualité pour tous, un des 17 objectifs de l'ONU.", True),
    ("Éducation", "RSA", "Recensement Scolaire Annuel",
     "Enquête annuelle sur les établissements scolaires (édition 2024 utilisée).", True),
    ("Éducation", "RGPH", "Recensement Général de la Population et de l'Habitat",
     "Source de la population (2022).", True),
    ("Éducation", "DIES", "Données Ouvertes sur les Indicateurs de l'Éducation Tous Secteurs au Togo",
     "Il couvre les indicateurs des secteurs préscolaire, primaire et secondaire. Le sigle provient probablement de Données Indicateurs Education Secteurs, bien que cette décomposition ne soit pas officiellement documentée. ", True),
    ("Éducation", "MEN", "Ministère de l'Éducation Nationale",
     "Ministère togolais chargé de l'éducation préscolaire, primaire et secondaire général. Producteur des données éducation publiées sur opendata.gouv.tg.", True),
    ("Éducation", "Préscolaire · Primaire · Collège · Lycée", "Cycles d'enseignement",
     "Maternelle, 1er cycle, 2nd cycle de l'enseignement secondaire, etc.", True),
    # --- Données ---
    ("Données", "opendata.gouv.tg / geodata.gouv.tg", "Portails de données ouvertes du Togo",
     "Sources principales des données statistiques et géographiques.", True),
    ("Données", "ISRI-TG", "Indices de Susceptibilité (FSI) et de Risque d'Inondation (FRI) au Togo",
     "C'est le nom du projet et du jeu de données. Il produit deux indices complémentaires : le FSI (Flood Susceptibility Index), qui mesure la prédisposition physique du territoire à l'inondation, et le FRI (Flood Risk Index), qui mesure le risque réel en combinant l'aléa, l'exposition et la vulnérabilité. ", True),
    ("Données", "CSV", "Comma-Separated Values",
     "Fichier tableau lisible dans Excel ; proposé en téléchargement.", True),
    ("Données", "N/D", "Non disponible",
     "La donnée n'existe pas dans les sources. Ce n'est pas un zéro.", True),
    ("Données", "Complétude", "Part des cases renseignées",
     "Indicateur de qualité des données (page « Qualité des données »).", True),
    # --- Lire les graphiques ---
    ("Graphiques", "Choroplèthe", "Carte colorée par valeur",
     "Plus la couleur est foncée, plus la valeur est forte. Gris = pas de donnée.", True),
    ("Graphiques", "Histogramme", "Barres par tranche de valeurs",
     "Montre combien de territoires se situent dans chaque tranche.", True),
    ("Graphiques", "Radar", "Graphique en toile d'araignée",
     "Chaque branche est un pilier ; la zone remplie résume le profil.", True),
    ("Graphiques", "Interpolation", "Estimation entre deux points connus",
     "Utilisée dans la simulation éducation entre les cibles 35, 40 et 50.", True),
]


# ────────────────────────────────────────────────────────────────
# 3. AIDE À LA DÉCISION — règles, stockage, page
# ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DECISION_LOG = DATA_DIR / "decisions_log.csv"
FEEDBACK_LOG = DATA_DIR / "feedback_app.csv"
DECISION_COLS = ["horodatage", "validateur", "fonction", "niveau", "territoire", "region",
                 "icv", "classe", "pilier", "urgence", "action_proposee", "decision",
                 "justification", "utilite_1_5", "commentaire_general", "poids_icv"]
FEEDBACK_COLS = ["horodatage", "auteur", "note_1_5", "categorie", "commentaire"]
DECISIONS = {"valide": "✅ Validée", "modif": "✏️ Validée avec modification",
             "etudier": "⏸ À étudier plus tard", "rejet": "❌ Rejetée"}

# 📘 Catalogue d'actions types par pilier : (action, détail, horizon).
#    Ce sont des PROPOSITIONS génériques à adapter au contexte local.
ACTIONS_CATALOGUE = {
    "sante": [
        ("Construire ou réhabiliter une formation sanitaire",
         "Cibler les localités les plus éloignées de toute formation sanitaire.", "12-24 mois"),
        ("Renforcer personnel et équipements des formations existantes",
         "Affecter du personnel qualifié, du matériel de base et des médicaments essentiels.", "6-12 mois"),
        ("Organiser des cliniques mobiles et des agents de santé communautaires",
         "Solution rapide en attendant de nouvelles infrastructures.", "3-6 mois")],
    "eau": [
        ("Réaliser de nouveaux forages et points d'eau",
         "Prioriser les zones sous la norme de 4 points d'eau pour 1 000 habitants.", "6-18 mois"),
        ("Réhabiliter les ouvrages en panne",
         "Remettre en service les points d'eau existants : souvent plus rapide et moins coûteux.", "3-9 mois"),
        ("Mettre en place un comité de gestion et de maintenance",
         "Assurer l'entretien et le financement durable des ouvrages.", "3-6 mois")],
    "alimentation": [
        ("Améliorer l'accès aux marchés",
         "Pistes rurales, petits marchés locaux, transport des produits.", "6-18 mois"),
        ("Constituer un stock tampon alimentaire",
         "Réserve locale pour les périodes de soudure.", "3-9 mois"),
        ("Suivre et stabiliser les prix des denrées de base",
         "Relevés réguliers des prix et information des producteurs.", "3-6 mois")],
    "inondation": [
        ("Intégrer l'aléa inondation au plan d'aménagement",
         "Interdire ou encadrer la construction en zone à risque.", "6-12 mois"),
        ("Curer et aménager drains et caniveaux",
         "Améliorer l'évacuation des eaux avant la saison des pluies.", "3-9 mois"),
        ("Installer une alerte précoce et des plans d'évacuation",
         "Informer la population et prévoir des sites de repli.", "3-6 mois")],
    "erosion": [
        ("Renforcer ou construire des ouvrages de protection côtière",
         "Épis, brise-lames ou rechargement de plage selon l'étude technique.", "12-36 mois"),
        ("Encadrer les constructions dans la bande à risque",
         "Définir une zone de retrait par rapport au trait de côte.", "6-12 mois"),
        ("Suivre régulièrement le trait de côte",
         "Mesures périodiques pour vérifier l'efficacité des actions.", "3-6 mois")],
}

# Pour chaque pilier : (colonne observée, unité, norme) — pour la rubrique « Pourquoi ? ».
NORM_INFO = {
    "sante": ("health_fac_per_10000", "formations / 10 000 hab.", "≥ 2, OMS"),
    "eau": ("water_pts_per1000", "points d'eau / 1 000 hab.", "≥ 4, Sphère"),
    "alimentation": ("food_markets_per_10000", "marchés / 10 000 hab.", "≥ 1, approximation"),
    "inondation": ("flood_fsi_max", "(FRI max)", "≤ 60"),
}


def _read_log(path: Path, cols: list[str]) -> pd.DataFrame:
    """Lit un journal CSV (+ lignes gardées en mémoire si l'écriture disque a échoué)."""
    frames = []
    try:
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            frames.append(df[cols])
    except Exception:
        pass
    mem = st.session_state.get(f"mem_{path.name}", [])
    if mem:
        frames.append(pd.DataFrame(mem, columns=cols))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def _append_log(path: Path, cols: list[str], rows: list[dict]) -> bool:
    """Ajoute des lignes à la fin du journal (jamais de modification : traçabilité).

    Retourne False si le disque est inaccessible ; les lignes sont alors gardées en
    mémoire pour la session et restent exportables."""
    try:
        DATA_DIR.mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=cols).to_csv(
            path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
        return True
    except Exception:
        st.session_state.setdefault(f"mem_{path.name}", []).extend(rows)
        return False


def build_recommendations(row) -> list[dict]:
    """Propose des actions à partir des scores de piliers (règles simples et visibles).

    📘 RÈGLES : on retient les piliers dont le score est ≥ 50 (3 au maximum). Score ≥ 75 =
       « Urgent » (3 actions), 50-74 = « Prioritaire » (2 actions). Si aucun pilier n'atteint 50,
       on propose 1 action de « Veille » sur le pilier le plus élevé."""
    scores = {}
    for p in PILLARS:
        k = f"score_{p}"
        if k in row.index and pd.notna(row[k]):
            scores[p] = float(row[k])
    if not scores:
        return []
    ordered = sorted(scores, key=scores.get, reverse=True)
    picked = [p for p in ordered if scores[p] >= 50][:3] or ordered[:1]
    recs = []
    for p in picked:
        s = scores[p]
        urg = "Urgent" if s >= 75 else ("Prioritaire" if s >= 50 else "Veille")
        n_act = 3 if urg == "Urgent" else 2 if urg == "Prioritaire" else 1
        preuve = f"Score {PILLAR_LABEL[p]} : {s:.0f}/100."
        if p in NORM_INFO and NORM_INFO[p][0] in row.index and pd.notna(row[NORM_INFO[p][0]]):
            col, unit, seuil = NORM_INFO[p]
            try:
                preuve += f" Valeur observée : {float(row[col]):.2f} {unit} (norme : {seuil})."
            except (TypeError, ValueError):
                pass
        for action, detail, horizon in ACTIONS_CATALOGUE[p][:n_act]:
            recs.append({"pilier": PILLAR_LABEL[p], "urgence": urg, "score": s,
                         "preuve": preuve, "action": action, "detail": detail,
                         "horizon": horizon})
    return recs


URGENCE_ICON = {"Urgent": "🔴 Urgent", "Prioritaire": "🟠 Prioritaire", "Veille": "🟢 Veille"}


def page_decision():
    hero("Aide à la décision",
         "Des propositions expliquées : vous décidez, l'application garde la trace.",
         chips=["Propositions expliquées", "Validation humaine", "Journal traçable"])
    if gdf.empty:
        box("alert", "Aucune donnée pour ce niveau.")
        return

    box("info", "Les propositions sont produites par des <b>règles simples et visibles</b> (scores des "
        "piliers, normes). Elles ne sont <b>jamais appliquées automatiquement</b> : pour chacune, vous "
        "validez, modifiez ou rejetez. Votre nom, la date et votre justification sont enregistrés.",
        "Vous décidez")

    tab1, tab2, tab3 = st.tabs(["1️⃣ Analyser & décider", "2️⃣ Journal des décisions",
                                "3️⃣ Retour d'expérience"])

    # ═══ ONGLET 1 — ANALYSER & DÉCIDER ═══════════════════════════
    with tab1:
        flash = st.session_state.pop("dec_flash", None)
        if flash:
            box(flash[0], flash[1])

        section("Qui valide ?", "pop")
        a, b = st.columns(2)
        with a:
            st.text_input("Votre nom (obligatoire)", key="dec_nom", placeholder="Prénom NOM")
        with b:
            st.text_input("Votre fonction", key="dec_fonction", placeholder="Ex. Responsable planification")

        section("Territoires à examiner en priorité", "alert")
        st.caption("👆 Cliquez une ligne pour préparer la décision de ce territoire.")
        top = gdf_view.sort_values("ICV", ascending=False).head(10)
        cols_p = [c for c in ["commune", "region", "ICV", "classe"] if c in top.columns]
        dfp = top[cols_p].round(1).rename(columns={"commune": "Territoire", "region": "Région",
                                                   "classe": "Classe"})
        picked = table_with_select(dfp, "dec_priority", "Territoire", height=300)
        if picked and picked != st.session_state.get("dec_last_pick"):
            st.session_state["dec_last_pick"] = picked
            st.session_state["dec_select"] = picked

        noms = sorted(gdf["commune"].dropna().unique())
        tgt = st.session_state.pop("decision_target", None)   # posé par la fiche territoriale
        if tgt in noms:
            st.session_state["dec_select"] = tgt
        choix = st.selectbox("Territoire à traiter", noms, key="dec_select")
        row = gdf[gdf["commune"] == choix].iloc[0]

        n_alert = sum(bool(row.get(c)) for c in ("alerte_sante", "alerte_eau", "alerte_alim",
                                                  "alerte_ino", "alerte_ero") if c in row.index)
        pop = (f"{int(row['population']):,}".replace(",", " ")
               if pd.notna(row.get("population")) else "N/D")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("ICV", f"{row['ICV']:.0f}" if pd.notna(row.get("ICV")) else "N/D",
                     str(row.get("classe", "")), "danger" if row.get("classe") == "Critique" else "warn",
                     icon="scale", pct=row.get("ICV", 0))
        with c2:
            kpi_card("Alertes", n_alert, "normes non respectées", "danger" if n_alert >= 3 else "warn",
                     icon="alert")
        with c3:
            kpi_card("Population", pop, "habitants", icon="pop")
        with c4:
            kpi_card("Fiche complète", "Ouvrir", "profil et radar", icon="fiche",
                     target="fiche", state={"fiche_target": choix}, hint="Voir la fiche")

        # Scores des piliers (pour comprendre d'où viennent les propositions)
        sc = [(PILLAR_LABEL[p], float(row[f"score_{p}"])) for p in PILLARS
              if f"score_{p}" in row.index and pd.notna(row[f"score_{p}"])]
        if sc:
            dsc = pd.DataFrame(sc, columns=["Pilier", "Score"]).sort_values("Score")
            fig = px.bar(dsc, x="Score", y="Pilier", orientation="h", color="Score",
                         color_continuous_scale="YlOrRd", range_color=[0, 100], text="Score")
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, xaxis=dict(range=[0, 110]))
            st.plotly_chart(_clean_fig(fig, 280, "calcul VIGIE·TOGO",
                                       title="Scores des 4 piliers ICV (100 = le plus défavorable)"),
                            width="stretch")

        recs = build_recommendations(row)
        section("Propositions — à vous de décider", "check")
        if not recs:
            box("info", "Aucune proposition possible : les scores des piliers sont absents pour ce territoire.")
        else:
            box("tip", "Pour chaque proposition, choisissez une décision. Une <b>justification</b> est "
                "obligatoire pour une modification ou un rejet. Les propositions laissées sans choix "
                "ne sont pas enregistrées.", "Mode d'emploi")

            slug = re.sub(r"\W+", "_", str(choix))
            kp = f"dec_{slug}_{st.session_state.get('dec_nonce', 0)}"   # clé unique, change après envoi
            with st.form(f"form_{kp}"):
                for i, r in enumerate(recs):
                    with st.container(border=True):
                        st.markdown(
                            f"**{i + 1}. {r['action']}** &nbsp; {URGENCE_ICON[r['urgence']]} · "
                            f"{r['pilier']} · horizon {r['horizon']}  \n{r['detail']}  \n"
                            f"<small>🔎 Pourquoi cette proposition ? {r['preuve']}</small>",
                            unsafe_allow_html=True)
                        st.radio("Votre décision", list(DECISIONS.values()), index=None,
                                 horizontal=True, key=f"{kp}_d{i}")
                        st.text_area("Justification (pour une modification, indiquez l'action retenue)",
                                     key=f"{kp}_j{i}", height=68)
                        st.radio("Cette proposition vous est-elle utile ? (1 = pas du tout, 5 = très utile)",
                                 [1, 2, 3, 4, 5], index=None, horizontal=True, key=f"{kp}_u{i}")

                st.markdown("**➕ Ajouter votre propre action** (facultatif)")
                cc1, cc2 = st.columns([3, 1])
                with cc1:
                    custom = st.text_input("Action que vous souhaitez ajouter", key=f"{kp}_custom")
                with cc2:
                    custom_p = st.selectbox("Pilier", PILLARS, format_func=lambda p: PILLAR_LABEL[p],
                                            key=f"{kp}_customp")
                comment = st.text_area("Commentaire général (contexte local, contraintes, sources…)",
                                       key=f"{kp}_comment", height=80)
                submitted = st.form_submit_button("💾 Enregistrer mes décisions", type="primary")

            if submitted:
                nom = (st.session_state.get("dec_nom") or "").strip()
                errs, rows = [], []
                if not nom:
                    errs.append("Indiquez votre nom (champ « Qui valide ? » plus haut).")
                horo = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                poids = _json.dumps({k: round(v, 3) for k, v in w.items()}, ensure_ascii=False)
                base = {"horodatage": horo, "validateur": nom,
                        "fonction": (st.session_state.get("dec_fonction") or "").strip(),
                        "niveau": niveau_label, "territoire": choix, "region": row.get("region", ""),
                        "icv": f"{row['ICV']:.1f}" if pd.notna(row.get("ICV")) else "",
                        "classe": row.get("classe", ""), "commentaire_general": (comment or "").strip(),
                        "poids_icv": poids}
                for i, r in enumerate(recs):
                    d = st.session_state.get(f"{kp}_d{i}")
                    if d is None:
                        continue
                    j = (st.session_state.get(f"{kp}_j{i}") or "").strip()
                    if d in (DECISIONS["modif"], DECISIONS["rejet"]) and not j:
                        errs.append(f"Proposition {i + 1} : une justification est obligatoire "
                                    "pour une modification ou un rejet.")
                    u = st.session_state.get(f"{kp}_u{i}")
                    rows.append({**base, "pilier": r["pilier"], "urgence": r["urgence"],
                                 "action_proposee": r["action"], "decision": d,
                                 "justification": j, "utilite_1_5": "" if u is None else str(u)})
                if (custom or "").strip():
                    rows.append({**base, "pilier": PILLAR_LABEL[custom_p], "urgence": "Ajoutée",
                                 "action_proposee": "Action ajoutée par l'utilisateur",
                                 "decision": DECISIONS["valide"], "justification": custom.strip(),
                                 "utilite_1_5": ""})
                if not rows and not errs:
                    errs.append("Aucune décision à enregistrer : choisissez au moins une option.")
                if errs:
                    for e in errs:
                        box("alert", e, "À corriger")
                else:
                    ok = _append_log(DECISION_LOG, DECISION_COLS, rows)
                    st.session_state["dec_nonce"] = st.session_state.get("dec_nonce", 0) + 1
                    msg = f"{len(rows)} décision(s) enregistrée(s) pour {choix}."
                    st.session_state["dec_flash"] = (
                        ("ok", msg) if ok else
                        ("tip", msg + " Le disque n'est pas accessible : elles sont gardées pour cette "
                                "session, pensez à télécharger le journal (onglet 2)."))
                    st.rerun()

    # ═══ ONGLET 2 — JOURNAL ═══════════════════════════════════════
    with tab2:
        section("Journal des décisions", "fiche")
        dfj = _read_log(DECISION_LOG, DECISION_COLS)
        box("info", "Le journal est en <b>ajout seul</b> : on n'efface rien, pour garder une trace fiable. "
            "Pour corriger une décision, enregistrez-en une nouvelle sur le même territoire.", "Traçabilité")
        if dfj.empty:
            box("tip", "Aucune décision enregistrée pour l'instant. Elles apparaîtront ici dès l'onglet 1 validé.")
        else:
            acc = dfj["decision"].isin([DECISIONS["valide"], DECISIONS["modif"]]).sum()
            rej = (dfj["decision"] == DECISIONS["rejet"]).sum()
            eti = (dfj["decision"] == DECISIONS["etudier"]).sum()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("Décisions", len(dfj), f"{dfj['territoire'].nunique()} territoire(s)", icon="fiche")
            with c2:
                kpi_card("Retenues", int(acc), f"{100 * acc / max(len(dfj), 1):.0f} % des propositions",
                         "ok", icon="check", pct=100 * acc / max(len(dfj), 1))
            with c3:
                kpi_card("À étudier", int(eti), "en attente", "warn", icon="info")
            with c4:
                kpi_card("Rejetées", int(rej), "avec justification", "danger" if rej else "", icon="alert")

            f1, f2 = st.columns(2)
            ft = f1.multiselect("Territoire", sorted(dfj["territoire"].unique()), key="jr_terr")
            fd = f2.multiselect("Décision", list(DECISIONS.values()), key="jr_dec")
            dv = dfj.copy()
            if ft:
                dv = dv[dv["territoire"].isin(ft)]
            if fd:
                dv = dv[dv["decision"].isin(fd)]
            dv = dv.sort_values("horodatage", ascending=False)
            show = dv[["horodatage", "territoire", "pilier", "action_proposee", "decision",
                       "justification", "validateur", "utilite_1_5"]].rename(columns={
                "horodatage": "Date", "territoire": "Territoire", "pilier": "Pilier",
                "action_proposee": "Action proposée", "decision": "Décision",
                "justification": "Justification", "validateur": "Validé par",
                "utilite_1_5": "Utilité (1-5)"})
            st.caption("👆 Cliquez une ligne pour ouvrir la fiche du territoire.")
            nm = table_with_select(show, "journal_table", "Territoire", height=380)
            if nm:
                open_fiche(nm, key="journal_open_fiche", label=f"Ouvrir la fiche de {nm} →")
            st.download_button("⬇ Télécharger le journal (CSV)",
                               data=dv.to_csv(index=False).encode("utf-8-sig"),
                               file_name="journal_decisions_vigie_togo.csv", mime="text/csv")

    # ═══ ONGLET 3 — RETOUR D'EXPÉRIENCE ═══════════════════════════
    with tab3:
        section("Ce que disent vos décisions", "analyse")
        dfj = _read_log(DECISION_LOG, DECISION_COLS)
        if dfj.empty:
            box("tip", "Les statistiques apparaîtront après les premières décisions.")
        else:
            dfj["utilite"] = pd.to_numeric(dfj["utilite_1_5"], errors="coerce")
            g = dfj.groupby(["pilier", "decision"]).size().reset_index(name="n")
            fig = px.bar(g, x="pilier", y="n", color="decision", barmode="stack",
                         labels={"n": "Nombre", "pilier": "", "decision": "Décision"})
            st.plotly_chart(_clean_fig(fig, 340, "journal des décisions",
                                       title="Décisions par pilier"), width="stretch")
            util = (dfj.groupby("pilier")["utilite"].agg(["mean", "count"]).round(2)
                    .rename(columns={"mean": "Utilité moyenne (1-5)", "count": "Notes"}).reset_index()
                    .rename(columns={"pilier": "Pilier"}))
            util = util[util["Notes"] > 0]
            if not util.empty:
                st.dataframe(util, width="stretch", hide_index=True)
            rej = dfj[dfj["decision"] == DECISIONS["rejet"]]
            if not rej.empty:
                section("Motifs de rejet : pour améliorer les propositions", "alert")
                st.dataframe(rej[["territoire", "pilier", "action_proposee", "justification",
                                  "validateur", "horodatage"]], width="stretch", hide_index=True)

        section("Votre avis sur l'application", "info")
        with st.form("app_feedback", clear_on_submit=True):
            auteur = st.text_input("Votre nom (facultatif)")
            note = st.radio("Note globale (1 = insatisfait, 5 = très satisfait)", [1, 2, 3, 4, 5],
                            index=None, horizontal=True)
            cat = st.selectbox("Sujet", ["Clarté des explications", "Qualité des données",
                                         "Fonctionnalité manquante", "Erreur ou bug", "Autre"])
            com = st.text_area("Votre commentaire", height=90)
            if st.form_submit_button("Envoyer mon avis", type="primary"):
                if not (com or "").strip() and note is None:
                    box("alert", "Donnez une note ou écrivez un commentaire.")
                else:
                    ok = _append_log(FEEDBACK_LOG, FEEDBACK_COLS, [{
                        "horodatage": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "auteur": (auteur or "").strip(), "note_1_5": "" if note is None else str(note),
                        "categorie": cat, "commentaire": (com or "").strip()}])
                    box("ok" if ok else "tip", "Merci pour votre retour !" if ok else
                        "Merci ! Le disque n'est pas accessible : votre avis est gardé pour cette session.")
        fb = _read_log(FEEDBACK_LOG, FEEDBACK_COLS)
        if not fb.empty:
            notes = pd.to_numeric(fb["note_1_5"], errors="coerce")
            st.caption(f"{len(fb)} avis reçu(s)" +
                       (f" · note moyenne {notes.mean():.1f}/5" if notes.notna().any() else ""))
            st.download_button("⬇ Télécharger les avis (CSV)", data=fb.to_csv(index=False).encode("utf-8-sig"),
                               file_name="avis_vigie_togo.csv", mime="text/csv")

# ────────────────────────────────────────────────────────────────
def page_methode():
    hero("Méthodologie & guide",
         "Tout comprendre : pages, sigles, calculs, normes, limites.",
         chips=["Guide pas à pas", "Glossaire", "Calculatrice ICV", "Limites"])

    tabs = st.tabs(["🚀 Démarrer", "🗂 Les pages une à une", "🧮 Calcul de l'ICV", "📏 Normes & seuils",
                    "🔤 Glossaire", "🗳 Processus de décision", "⚠ Limites & bonnes pratiques"])

    # ── 1. Démarrer ──────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
### À quoi sert VIGIE·TOGO 360 ?

C'est un **atlas de la vulnérabilité** du Togo. Il montre, territoire par territoire, où les
populations sont les plus exposées (santé, eau, alimentation, inondation, érosion côtière) et il aide à
**décider où agir en premier**.

### Bien démarrer en 5 gestes

1. **Choisissez l'échelle** (régions, préfectures, communes, cantons) dans la barre latérale à gauche.
2. **Partez de l'Accueil** : chaque tuile et chaque chiffre est cliquable.
3. **Explorez** : sur une carte, un graphique ou un tableau, cliquez un territoire puis « Ouvrir la fiche ».
4. **Ajustez les poids** de l'ICV (barre latérale) si un pilier compte plus pour votre décision.
5. **Décidez** dans « Aide à la décision » : l'application propose, vous validez.

### Comment lire les couleurs
""")
        st.markdown(
            f'<span class="badge badge-critique"><span class="dot"></span>Critique</span> ICV &gt; 75 &nbsp; '
            f'<span class="badge badge-eleve"><span class="dot"></span>Élevée</span> 50 à 75 &nbsp; '
            f'<span class="badge badge-modere"><span class="dot"></span>Modérée</span> 25 à 50 &nbsp; '
            f'<span class="badge badge-faible"><span class="dot"></span>Faible</span> &lt; 25',
            unsafe_allow_html=True)
        st.markdown("""
- **Carte KPI à bordure verte** : situation correcte ou information neutre. **Orange** : à surveiller. **Rouge** (parfois qui pulse) : critique.
- **Carte** : plus la teinte est foncée, plus la valeur est forte ; gris = donnée absente.
- **« Voir le détail → »** en bas d'une carte : elle est cliquable et ouvre une autre page.

### Les outils de la barre latérale
- **Maille territoriale** : l'échelle d'analyse.  **Région** : limite l'affichage à une région.
- **Mode citoyen** : fiche simplifiée à imprimer.  **Pondération ICV** : l'importance de chaque pilier.
- **❓ Aide rapide** : rappel des gestes essentiels.

### Imprimer
Bouton « Imprimer » de la fiche, ou **Ctrl+P** (Windows) / **Cmd+P** (Mac). Les menus sont masqués à l'impression.
""")

    # ── 2. Les pages une à une ───────────────────────────────
    with tabs[1]:
        st.markdown("Chaque page est expliquée ci-dessous : à quoi elle sert, comment la lire, comment agir.")
        groupes: dict[str, list] = {}
        for k, _f, titre, _m, grp, _s, desc in PAGES_DEF:
            groupes.setdefault(grp, []).append((k, titre, desc))
        for grp, items in groupes.items():
            section(grp, "map")
            for k, titre, desc in items:
                with st.expander(f"{titre} — {desc}"):
                    if k in PAGES_HELP:
                        _render_help(PAGES_HELP[k])
                    elif k == "accueil":
                        st.markdown("Point de départ : alerte sur les territoires critiques, tuiles de "
                                    "                                    navigation, chiffres clés, 4 piliers ICV, top 5 des territoires les plus "
                                    "vulnérables et répartition par classe. Tout est cliquable.")
                    elif k == "methode":
                        st.markdown("La page que vous lisez : guide, calculs, normes, glossaire, limites.")
                    if k in NAV_PAGES and k not in ("methode",):
                        if st.button(f"Ouvrir « {titre} » →", key=f"guide_open_{k}"):
                            st.switch_page(NAV_PAGES[k])

    # ── 3. Calcul de l'ICV ───────────────────────────────────
    with tabs[2]:
        st.markdown("""
### Les 4 piliers ICV

Chaque pilier reçoit un **score de 0 à 100**, où **100 = situation la plus défavorable**.

| Pilier | Ce qu'on mesure | Plus le score est haut… |
|---|---|---|
| **Santé** | Habitants par formation sanitaire ; formations pour 10 000 habitants | …moins la population a accès aux soins |
| **Eau** | Points d'eau pour 1 000 habitants ; densité de points | …moins la population a accès à l'eau |
| **Alimentation** | Marchés pour 10 000 habitants ; densité de marchés | …moins la population a accès aux marchés |
| **Inondation** | Indice FRI ; FSI maximum | …plus le risque d'inondation est fort |
| **Érosion côtière** | EPR moyen et maximum | …plus le recul du littoral est rapide |

### La formule

**ICV = Σ (poids × score du pilier)**, avec la somme des poids égale à 1.
Les poids sont de 0,25 par défaut avant normalisation ; vous les changez dans la barre latérale.
Le calcul détaillé des scores de piliers est réalisé par le module `icv.py`.

### La classe

Faible (ICV < 25) · Modérée (25-50) · Élevée (50-75) · Critique (> 75).
""")
        section("Calculatrice pédagogique : refaites le calcul vous-même", "scale")
        st.caption("Illustration de la formule avec vos propres valeurs. Elle ne modifie pas les données de l'application.")
        cols = st.columns(4)
        sc_in, w_in = {}, {}
        for col, p in zip(cols, PILLARS):
            with col:
                sc_in[p] = st.slider(f"Score {PILLAR_LABEL[p]}", 0, 100, 50, key=f"calc_s_{p}")
                w_in[p] = st.slider(f"Poids {PILLAR_LABEL[p]}", 0.0, 1.0, 0.25, 0.05, key=f"calc_w_{p}")
        tot = sum(w_in.values()) or 1.0
        icv_c = sum(sc_in[p] * w_in[p] / tot for p in PILLARS)
        classe_c = ("Faible" if icv_c < 25 else "Modérée" if icv_c < 50 else "Élevée" if icv_c < 75 else "Critique")
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("ICV calculé", f"{icv_c:.1f}", "0 – 100", "danger" if classe_c == "Critique" else "warn",
                     icon="scale", pct=icv_c)
        with c2:
            st.markdown(f"<div style='padding-top:1.4rem'>Classe : {badge_classe(classe_c)}</div>",
                        unsafe_allow_html=True)
        st.caption("Détail : " + " + ".join(f"{w_in[p] / tot:.2f} × {sc_in[p]}" for p in PILLARS) + f" = {icv_c:.1f}")

    # ── 4. Normes & seuils ───────────────────────────────────
    with tabs[3]:
        st.markdown("""
### Normes utilisées pour déclencher une alerte

| Domaine | Norme de référence | Seuil | Alerte si… |
|---|---|---|---|
| Santé | OMS | ≥ 2 formations sanitaires / 10 000 hab. | en dessous |
| Eau | Sphère | ≥ 4 points d'eau / 1 000 hab. | en dessous |
| Alimentation | Approximation de travail | ≥ 1 marché / 10 000 hab. | en dessous |
| Inondation | Indice FRI | FRI ≤ 60 | au-dessus de 60 |
| Éducation, encadrement | UNESCO | ≤ 40 élèves par enseignant | au-dessus |
| Éducation, parité | Parité filles/garçons | IPS = 1,00 | < 0,95 inégalité ; < 0,80 critique |

### Seuils de lecture propres à l'application
- **Classes ICV** : 25 · 50 · 75.  **Hot-spot** : 3 alertes ou plus.
- **Matrice de priorisation** : lignes à 60 (déficit de services) et 55 (exposition inondation).
- **Risque de décrochage** : « critique » à partir de 60/100.
- **Propositions d'action** : un pilier est retenu à partir de 50/100 ; « Urgent » à partir de 75/100.

Ces seuils sont des **conventions de lecture** : modifiables dans le code si votre contexte l'exige.
""")

    # ── 5. Glossaire ─────────────────────────────────────────
    with tabs[4]:
        st.markdown("Tous les sigles et termes de l'application. Tapez un mot pour filtrer.")
        c1, c2 = st.columns([2, 1])
        q = c1.text_input("🔎 Rechercher", key="gloss_q", placeholder="Ex. IPS, eau, FCFA…")
        cats = sorted({g[0] for g in GLOSSAIRE})
        cat = c2.selectbox("Catégorie", ["Toutes"] + cats, key="gloss_cat")
        rows = []
        for entry in GLOSSAIRE:
            # Certaines entrées peuvent avoir 4 ou 5 éléments → on normalise sans planter
            if len(entry) == 5:
                c, sigle, dev, expl, sure = entry
            elif len(entry) == 4:
                c, sigle, dev, sure = entry
                expl = ""
            else:
                continue
            if cat != "Toutes" and c != cat:
                continue
            if q and q.lower() not in f"{sigle} {dev} {expl}".lower():
                continue
            rows.append({"Sigle / terme": sigle, "Signification": dev, "Explication": expl,
                         "Catégorie": c, "Statut": "✔ Confirmé" if sure else "À confirmer"})
        st.caption(f"{len(rows)} entrée(s)")
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=520,
                         column_config={"Explication": st.column_config.TextColumn(width="large"),
                                        "Signification": st.column_config.TextColumn(width="medium")})
        else:
            box("tip", "Aucun résultat. Essayez un autre mot ou choisissez « Toutes ».")
        box("info", "Les entrées « À confirmer » sont des sigles repris des fichiers sources dont la "
            "signification exacte n'est pas documentée : demandez-la au producteur des données.", "Transparence")

    # ── 6. Processus de décision ─────────────────────────────
    with tabs[5]:
        st.markdown("""
### L'application propose, l'utilisateur décide

1. **Repérer** : le tableau de bord, l'analyse et les normes montrent où se concentrent les difficultés.
2. **Examiner** : choisissez un territoire dans « Aide à la décision ». Les scores des piliers et les valeurs
   observées expliquent d'où viennent les propositions.
3. **Décider** : pour chaque proposition, choisissez
   - ✅ **Validée** : vous la retenez telle quelle ;
   - ✏️ **Validée avec modification** : vous la retenez en la corrigeant (justification obligatoire) ;
   - ⏸ **À étudier plus tard** : en attente d'informations ;
   - ❌ **Rejetée** : vous ne la retenez pas (justification obligatoire).
4. **Noter** l'utilité de la proposition (1 à 5) : ce retour sert à améliorer les règles.
5. **Enregistrer** : votre nom, la date, le contexte (ICV, classe, poids utilisés) et votre justification sont conservés.
6. **Suivre** : le journal des décisions est consultable et téléchargeable en CSV.

### Pourquoi ce fonctionnement ?
- **Responsabilité** : une personne identifiée assume chaque décision.
- **Transparence** : chaque proposition montre les chiffres qui la justifient.
- **Amélioration** : les rejets motivés et les notes permettent d'ajuster les règles.

Le journal est enregistré dans `data/decisions_log.csv`, à côté de `app.py`. Les avis sur l'application
vont dans `data/feedback_app.csv`.
""")
        if st.button("🗳 Ouvrir l'aide à la décision →", key="meth_open_decision", type="primary"):
            st.switch_page(NAV_PAGES["decision"])

    # ── 7. Limites & bonnes pratiques ────────────────────────
    with tabs[6]:
        st.markdown("""
### Ce que l'application ne fait pas
- L'ICV est un indicateur **relatif** : il classe les territoires entre eux, il ne mesure pas la pauvreté ni un niveau de vie.
- Les **propositions** sont génériques. Elles ne remplacent ni une étude technique ni une consultation locale, et ne donnent ni coût ni calendrier précis.
- Les **simulations** testent des hypothèses ; elles ne prévoient pas l'avenir.
- Une **corrélation** (deux phénomènes qui évoluent ensemble) n'est pas une **cause**.

### Âge et qualité des données
- Prix PAM : **2001 à 2022**. Éducation : de **2019 à 2024** selon l'indicateur (année affichée sous chaque KPI).
- **Mode démonstration** : si la bannière rouge apparaît, les chiffres sont fictifs.
- **N/D** : donnée absente (pas zéro). Les moyennes ignorent ces cases.
- Vérifiez la page « Qualité des données » avant de vous appuyer sur un classement.
- Les valeurs marquées **« À confirmer »** ne doivent pas être publiées sans vérification.

### Bonnes pratiques
- Comparez toujours à l'échelle choisie (régions avec régions, communes avec communes).
- Testez plusieurs pondérations : un territoire qui reste critique quelle que soit la pondération est une priorité solide.
- Croisez avec la connaissance du terrain avant de décider.
- Notez le contexte dans le commentaire de chaque décision.

### Citer cet outil
« VIGIE·TOGO 360 — Observatoire de la Vulnérabilité et de la Résilience Territoriale », avec la date de consultation et les sources listées dans « Sources & licences ».
""")


# ════════════════════════════════════════════════════════════════
# 7. NAVIGATION (menu regroupé en 3 familles)
# ════════════════════════════════════════════════════════════════
# 📘 UNE seule liste décrit toutes les pages :
#    (clé, fonction, titre du menu, icône du menu, famille, icône tuile, description)
PAGES_DEF = [
    # ── Famille 1 : Vue d'ensemble ──
    ("accueil",   page_accueil,           "Accueil",                 ":material/home:",         "Vue d'ensemble",
     "home",      "Point de départ de l'atlas"),
    ("dashboard", page_dashboard,         "Tableau de bord",         ":material/dashboard:",    "Vue d'ensemble",
     "dashboard", "Carte, classement et distribution de l'ICV"),
    ("analyse",   page_analyse,           "Analyse territoriale",    ":material/insights:",     "Vue d'ensemble",
     "analyse",   "Matrice de priorisation et hot-spots"),
    ("scenarios", page_scenarios,         "Scénarios",               ":material/auto_awesome:", "Vue d'ensemble",
     "scenario",  "Simuler l'effet d'une amélioration"),
    ("decision",  page_decision,          "Aide à la décision",      ":material/how_to_vote:",  "Vue d'ensemble",
     "scale",     "Propositions expliquées, validées par vous"),
    # ── Famille 2 : Analyse thématique ──
    ("education", page_education,         "Éducation",               ":material/school:",       "Analyse thématique",
     "fiche",     "Décrochage, inégalités, écoles"),
    #("denrees",   page_denrees,           "Denrées & Marchés",       ":material/agriculture:",  "Analyse thématique",
     #"alimentation", "Prix et accès aux marchés"),
    ("projets",   page_projets,           "Projets & investissements", ":material/construction:", "Analyse thématique",
     "projets",   "Chantiers, réalisation, ZAAP"),
    ("catalogue", page_catalogue_projets, "Tous les projets",        ":material/list:",         "Analyse thématique",
     "projets",   "Catalogue complet, tous secteurs"),
    ("normes",    page_normes,            "Diagnostic normatif",     ":material/rule:",         "Analyse thématique",
     "check",     "Écarts aux normes OMS, Sphère, IPC"),
    # ── Famille 3 : Données & territoire ──
    ("fiche",     page_fiche,             "Fiche territoriale",      ":material/description:",  "Données & territoire",
     "fiche",     "Profil détaillé d'un territoire"),
    ("prix",      page_prix,              "Prix & prévisions",       ":material/trending_up:",  "Données & territoire",
     "prix",      "Évolution des prix PAM / WFP"),
    ("qualite",   page_qualite,           "Qualité des données",     ":material/data_check:",   "Données & territoire",
     "qualite",   "Complétude et audit"),
    ("sources",   page_sources,           "Sources & licences",      ":material/menu_book:",    "Données & territoire",
     "sources",   "Traçabilité et licences"),
    ("methode",   page_methode,           "Méthodologie & guide",    ":material/functions:",    "Données & territoire",
     "methode",   "Guide complet, sigles, normes, limites"),
]

PAGE_META: dict[str, dict] = {}
pages: dict[str, list] = {}
for _key, _fn, _title, _mat, _group, _svg_name, _desc in PAGES_DEF:
    # La page d'accueil est la page par défaut (pas d'url_path personnalisé).
    _page = (st.Page(_fn, title=_title, icon=_mat, default=True) if _key == "accueil"
             else st.Page(_fn, title=_title, icon=_mat, url_path=_key))
    NAV_PAGES[_key] = _page
    PAGE_TITLES[_key] = _title
    PAGE_META[_key] = {"title": _title, "svg": _svg_name, "desc": _desc}
    pages.setdefault(_group, []).append(_page)

nav = st.navigation(pages)
CURRENT["key"] = next((k for k, t in PAGE_TITLES.items() if t == nav.title), None)
nav.run()

# ════════════════════════════════════════════════════════════════
# 8. PIED DE PAGE
# ════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="app-footer"><strong>{APP_NAME}</strong> — {APP_TAGLINE}<br>'
    f"Données : opendata.gouv.tg · geodata.gouv.tg · PAM/WFP · "
    f"Généré le {datetime.now():%d/%m/%Y}</div>",
    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 📘 BUGS CORRIGÉS PAR RAPPORT À LA VERSION 3.1
# ════════════════════════════════════════════════════════════════
#  1. Niveaux « Régions » et « Préfectures » : la clé accentuée ("régions")
#     ne correspondait pas aux données → dictionnaire NIVEAUX explicite.
#  2. Cache de get_data jamais invalidé (paramètre `_v` ignoré par Streamlit).
#  3. Cache ICV : plus d'aller-retour JSON coûteux, mémorisation directe.
#  4. KPI de l'accueil : mélangeaient régions + préfectures + communes + cantons
#     (doubles comptes) → ils reflètent maintenant le niveau choisi.
#  5. Impression : iframe HTML remplacée par components.html (plus fiable).
#  6. Annotations Plotly : `font(weight=...)` remplacé par <b>…</b> (compatible).
#  7. Cartes : champs d'infobulle absents ignorés au lieu de planter.
#  8. CSS global sur div/span supprimé (cassait le thème sombre et les encadrés).