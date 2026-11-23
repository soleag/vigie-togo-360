"""Page Denrées & Marchés pour VIGIE·TOGO 360."""
import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from streamlit_folium import st_folium


def render(hero, kpi_card, section, box, folium_map, _clean_fig,
           levels, prices, communes):
    """Fonction principale appelée depuis app.py.
    Tous les composants UI sont passés en paramètres pour éviter les imports circulaires."""

    hero("Denrées & Marchés",
         "Toutes les cultures · Marchés géolocalisés · Sécurité alimentaire.",
         chips=["10 cultures", "1 078 marchés", "IPC 2024"])

    st.markdown(
        '<div style="background:#fffbeb; border-left:4px solid #d97706; '
        'padding:10px 14px; border-radius:8px; margin-bottom:1rem; '
        'font-size:.83rem; color:#78350f;">'
        '<b>Fraîcheur des données</b> — '
        'Marchés : décembre 2024 · Bilan vivrier : 2014-2018 · '
        'Prix WFP : 2001-2022 · IPC : mars 2024'
        '</div>',
        unsafe_allow_html=True,
    )

    bilan = levels.get("_bilan_vivrier", pd.DataFrame())
    ipc = levels.get("_ipc", pd.DataFrame())

    # ── KPI globaux ────────────────────────────────────────
    section("Vue d'ensemble", "dashboard")
    n_cultures = bilan["culture"].nunique() if not bilan.empty else 0
    n_marches = int(communes["nb_marches"].sum()) if "nb_marches" in communes else 0
    annee_max = int(bilan["annee"].max()) if not bilan.empty else 0
    n_regions = bilan["region"].nunique() if not bilan.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Cultures suivies", f"{n_cultures}",
                 "céréales, tubercules, légumineuses",
                 icon="alimentation", pct=100)
    with c2:
        kpi_card("Marchés recensés", f"{n_marches}",
                 "sur 117 communes", icon="map", pct=100)
    with c3:
        kpi_card("Régions couvertes", f"{n_regions}",
                 "bilan vivrier", icon="map", pct=100*n_regions/5 if n_regions else 0)
    with c4:
        kpi_card("Dernière campagne", f"{annee_max}" if annee_max else "N/D",
                 "bilan vivrier", icon="alimentation", pct=100)

    # ── Onglets ────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Productions", "Marchés", "Prix WFP", "Sécurité alimentaire",
    ])

    # ═══════════════════════════════════════════════════════
    # TAB 1 — PRODUCTIONS
    # ═══════════════════════════════════════════════════════
    with tab1:
        if bilan.empty:
            box("alert", "Bilan vivrier non disponible.")
        else:
            section("Productions par culture et région", "alimentation")

            c1, c2 = st.columns(2)
            with c1:
                cultures = sorted(bilan["culture"].dropna().unique())
                culture_sel = st.selectbox("Culture", cultures, key="den_cult")
            with c2:
                annees = sorted(bilan["annee"].dropna().unique(), reverse=True)
                annee_sel = st.selectbox("Année", annees, key="den_annee")

            df_cult = bilan[(bilan["culture"] == culture_sel) &
                            (bilan["annee"] == annee_sel)]

            if not df_cult.empty:
                col1, col2 = st.columns([2, 1])
                with col1:
                    regions_geo = levels.get("regions", pd.DataFrame()).copy()
                    if not regions_geo.empty:
                        df_map = regions_geo.merge(
                            df_cult[["region", "production"]],
                            left_on="commune", right_on="region", how="left",
                        )
                        if df_map["production"].notna().any():
                            m = folium_map(df_map, "production",
                                          f"Production {culture_sel} (t)")
                            st_folium(m, height=420, width="stretch")
                with col2:
                    st.markdown(f"**{culture_sel} en {annee_sel}**")
                    for _, r in df_cult.sort_values("production", ascending=False).iterrows():
                        st.metric(r["region"],
                                  f"{int(r['production']):,} t".replace(",", " "))

            section("Évolution de la production", "prix")
            df_evol = bilan[bilan["culture"] == culture_sel] \
                .groupby("annee")["production"].sum().reset_index()
            if not df_evol.empty:
                fig = px.line(df_evol, x="annee", y="production", markers=True,
                              labels={"annee": "Année",
                                      "production": "Production (tonnes)"})
                fig.update_traces(line_color="#006a4e", line_width=3)
                st.plotly_chart(
                    _clean_fig(fig, 350, "Bilan vivrier / opendata.gouv.tg",
                               title=f"Évolution {culture_sel}",
                               x_title="Année",
                               y_title="Production (tonnes)"),
                    width="stretch")

    # ═══════════════════════════════════════════════════════
    # TAB 2 — MARCHÉS
    # ═══════════════════════════════════════════════════════
    with tab2:
        section("Marchés géolocalisés", "map")
        try:
            marches_path = None
            for d in ("data/essentiel", "data", "data/_archive"):
                matches = sorted(Path(d).glob("alim_marches__*.csv*"))
                if matches:
                    marches_path = matches[0]
                    break
            if marches_path:
                df_m = pd.read_csv(marches_path, low_memory=False)
                regions_m = sorted(df_m["region_nom_bdd"].dropna().unique())
                region_m = st.selectbox("Filtrer par région",
                                         ["Toutes"] + regions_m,
                                         key="den_marq_region")
                if region_m != "Toutes":
                    df_m = df_m[df_m["region_nom_bdd"] == region_m]
                st.caption(f"**{len(df_m)} marchés** affichés")
                m = folium.Map(location=[8.5, 1.0], zoom_start=7,
                              tiles="OpenStreetMap", control_scale=True)
                for _, r in df_m.iterrows():
                    try:
                        from shapely import wkt
                        geom = wkt.loads(r["geometry"]) if isinstance(r["geometry"], str) else None
                        if geom:
                            folium.CircleMarker(
                                location=[geom.y, geom.x],
                                radius=4, color="#006a4e", weight=1,
                                fill=True, fillColor="#006a4e", fillOpacity=0.7,
                                popup=f"<b>{r.get('marche_nom', 'N/D')}</b>",
                                tooltip=r.get("marche_nom", ""),
                            ).add_to(m)
                    except Exception:
                        pass
                st_folium(m, height=450, width="stretch", key="den_marq_map")
        except Exception as e:
            box("alert", f"Erreur chargement marchés : {e}")

    # ═══════════════════════════════════════════════════════
    # TAB 3 — PRIX WFP
    # ═══════════════════════════════════════════════════════
    with tab3:
        if prices.empty:
            box("alert", "Prix WFP non disponibles.")
        else:
            section("Prix des denrées (WFP/PAM)", "prix")
            st.warning(
                "Attention : les données de prix WFP s'arretent en juin 2022. "
                "Il n'y a pas de mise a jour recente. Les prix sont indicatifs "
                "pour les tendances historiques."
            )
            c1, c2 = st.columns([1, 2])
            with c1:
                denrees = sorted(prices["denree"].dropna().unique())
                denree_sel = st.selectbox("Denrée", denrees, key="den_prix_denree")
            with c2:
                marches_p = sorted(prices["marche"].dropna().unique())
                marches_sel = st.multiselect(
                    "Marchés", marches_p,
                    default=marches_p[:4] if len(marches_p) > 4 else marches_p,
                    key="den_prix_marches",
                )
            sub = prices[(prices["denree"] == denree_sel) &
                        (prices["marche"].isin(marches_sel))]
            if not sub.empty:
                fig = px.line(sub, x="date", y="prix_fcfa_kg", color="marche",
                              markers=True,
                              labels={"prix_fcfa_kg": "Prix (FCFA/kg)",
                                      "date": "", "marche": "Marché"})
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(
                    _clean_fig(fig, 450, "PAM / WFP · opendata.gouv.tg",
                               title=f"Prix {denree_sel}",
                               x_title="Année", y_title="Prix (FCFA/kg)"),
                    width="stretch")
                c1, c2, c3 = st.columns(3)
                with c1:
                    kpi_card("Prix moyen",
                             f"{sub['prix_fcfa_kg'].mean():.0f} FCFA",
                             "par kg", icon="prix")
                with c2:
                    kpi_card("Min", f"{sub['prix_fcfa_kg'].min():.0f} FCFA",
                             "par kg", icon="prix")
                with c3:
                    kpi_card("Max", f"{sub['prix_fcfa_kg'].max():.0f} FCFA",
                             "par kg", icon="prix")

    # ═══════════════════════════════════════════════════════
    # TAB 4 — SÉCURITÉ ALIMENTAIRE
    # ═══════════════════════════════════════════════════════
    with tab4:
        if ipc.empty:
            box("alert", "Données IPC non disponibles.")
        else:
            section("Sécurité alimentaire (IPC mars 2024)", "alert")
            ipc_valid = ipc[ipc["phase"].isin(["1", "2", "3", "3+", "4", "5"])].copy()
            phases = sorted(ipc_valid["phase"].unique())
            phase_sel = st.selectbox("Phase IPC", phases, key="den_ipc_phase")
            df_phase = ipc_valid[ipc_valid["phase"] == phase_sel]
            if not df_phase.empty:
                st.markdown(f"**Population en phase {phase_sel}**")
                regions_geo = levels.get("regions", pd.DataFrame()).copy()
                if not regions_geo.empty and "region" in df_phase.columns:
                    agg = df_phase.groupby("region")["population"].sum().reset_index()
                    df_map = regions_geo.merge(agg, left_on="commune",
                                                right_on="region", how="left")
                    if df_map["population"].notna().any():
                        m = folium_map(df_map, "population",
                                      f"Population phase {phase_sel}")
                        st_folium(m, height=450, width="stretch")
                display_cols = ["region", "phase", "population"]
                if "pct" in df_phase.columns:
                    display_cols.append("pct")
                df_show = df_phase[display_cols].copy()
                df_show = df_show.rename(columns={
                    "region": "Zone", "phase": "Phase",
                    "population": "Population", "pct": "% région",
                })
                st.dataframe(df_show, width="stretch", hide_index=True)
