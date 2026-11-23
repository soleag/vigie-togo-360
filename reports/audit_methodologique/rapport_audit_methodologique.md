# Rapport d’audit méthodologique — VIGIE·TOGO 360

## Portée et reproductibilité

- Audit en lecture seule: aucune modification de `data_loader.py`, `icv.py`, des données brutes ou du tableau de bord.
- Chargement effectué avec `load_data()` puis calcul des piliers et de la double vulnérabilité existants.
- Livrables générés dans `reports/audit_methodologique/`.
- Mode chargement détecté: DONNEES LOCALES.

## 1. Faits vérifiés

- Nombre de cantons chargés: 388.
- Cantons identifiés en double peine aux seuils 60/30: 250.
- Cantons avec données insuffisantes: 0; non identifiés: 138; valides: 388.
- Les statistiques brutes, quantiles, corrélations, superficies et érosion sont détaillés dans les CSV associés.

- Corrélations Spearman fortes confirmées:
  - regions — sante_per_habitant_vs_densite: rho=-0.900, n=5.

## 2. Anomalies confirmées

- La valeur EPR/LRR `9999.99` est une sentinelle récurrente dans le GeoJSON brut.
- Après exclusion de cette sentinelle, l’EPR brut contient 474 valeurs interprétables.
- Moyenne EPR brute incluant les sentinelles: 1983.799 m/an.
- Moyenne EPR après exclusion: 5.119 m/an.
- Les superficies EPSG:3857 sont déformées; l’UTM zone 31N (EPSG:32631) constitue la comparaison équivalente locale.
- Superficies cantons: écart médian -2.609%, moyen -2.773%, plage [-4.123%, -1.772%].
- Superficies communes: écart médian -2.426%, moyen -2.666%, plage [-4.119%, -1.772%].
- Superficies prefectures: écart médian -2.599%, moyen -2.759%, plage [-4.108%, -1.792%].
- Superficies regions: écart médian -2.846%, moyen -2.856%, plage [-3.876%, -1.917%].

## 3. Hypothèses à vérifier

- Vérifier que `9999.99` est bien le seul code sentinelle documenté par le producteur.
- Obtenir une métadonnée officielle explicitant négatif=recul et positif=accrétion, et les bornes plausibles d’EPR/LRR.
- Vérifier si les territoires sans équipements représentent une absence réelle ou une couverture/source incomplète.
- Vérifier la pertinence du seuil 30 pour le score d’inondation agrégé; il n’est pas modifié par cet audit.

## 4. Recommandations

- Exclure toute valeur EPR/LRR égale à `9999.99` avant toute agrégation ou normalisation.
- Ne pas intégrer `erosion_epr_mean` à l’ICV tant que la sentinelle, la convention de signe et le périmètre littoral ne sont pas formellement validés.
- Utiliser une projection équivalente locale ou un calcul géodésique pour les futures densités; conserver EPSG:3857 inchangé tant que cette évolution n’est pas validée.
- Examiner les corrélations fortes sans supprimer automatiquement des indicateurs.
- Conserver les seuils 60/30 comme résultats de comparaison jusqu’à une décision métier documentée.
- Documenter séparément absence de données, absence d’équipement et zéro réel.

## 5. Limites méthodologiques

- EPSG:32631 est un choix de comparaison UTM adapté au Togo; il ne remplace pas une validation officielle du référentiel.
- Les extrêmes sont identifiés par la règle IQR 1.5, une méthode descriptive sans valeur normative.
- Spearman mesure une relation monotone, pas une causalité ni une redondance définitive.
- Les scores min-max dépendent du périmètre territorial chargé.
- L’agrégation FRI moyenne et FSI maximale reste celle du pipeline existant.
- Cet audit ne juge pas les pondérations ni ne valide les normes OMS, Sphère ou approximations.

## Résultats clés de la double vulnérabilité

- Résultat reproduit: **oui**.
- Identifiés: 250/388; valides: 388; données insuffisantes: 0; non identifiés: 138.
- `score_services`: disponible pour 388 cantons, médiane=82.027, moyenne=81.710.
- `score_inondation`: disponible pour 388 cantons, médiane=32.283, moyenne=39.852.
- Couverture des trois piliers de services — exactement 0 pilier(s): 0 cantons.
- Couverture des trois piliers de services — exactement 1 pilier(s): 0 cantons.
- Couverture des trois piliers de services — exactement 2 pilier(s): 0 cantons.
- Couverture des trois piliers de services — exactement 3 pilier(s): 388 cantons.
