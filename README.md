# VIGIE·TOGO 360 — dossier léger du Data Challenge

Ce dossier autonome regroupe le tableau de bord fonctionnel, ses données indispensables, les résultats vérifiés et les livrables de soumission. Les données tabulaires CSV sont conservées en `.csv.gz` pour rester sous la limite de 15 Mo; pandas les lit directement sans étape de décompression.

## Lancer le tableau de bord

Python 3.11+ est recommandé. Depuis ce dossier :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Le tableau de bord couvre les régions, préfectures, communes et cantons. L'ICV utilise quatre piliers actifs : santé, eau, alimentation et inondation. La section Éducation traite trois défis : anticiper le décrochage scolaire par région, mesurer les inégalités filles-garçons, et éclairer la répartition des écoles et des enseignants.

## Résultats verrouillés

Les résultats calculés à l'échelle cantonale et leurs limites sont documentés dans `reports/finalisation/resultats_final_verrouilles.md`. Les seuils 60/30, 70/50 et 75/55 sont analytiques, non officiels. Les données manquantes ne valent pas zéro; l'érosion côtière n'entre pas dans l'ICV.

## Fichiers

- `app.py`, `data_loader.py`, `icv.py`, `pages_denrees.py` : application et calculs.
- `data/essentiel/` : données nécessaires au chargement local; CSV compressés en gzip.
- La matrice de priorisation offre une carte colorée par zone et une liste des urgences; la vue à bulles reste accessible. Les couleurs de la carte reprennent les repères visuels 60/55, distincts des seuils analytiques 60/30.
- `config/sources_registry.json` et `SOURCES_DATASETS.md` : registre des sources.
- `tests/test_audit_p0.py` : tests de non-régression.
- `reports/` : audit méthodologique, résultats et limites.
- `livrables/` : présentation PowerPoint et fichier des sources.

Les détails méthodologiques complets figurent dans `reports/audit_methodologique/rapport_audit_methodologique.md` et `reports/finalisation/limites_methodologiques.md`.

## Vérification

```powershell
python -m pytest tests/test_audit_p0.py -q
python -m compileall -q app.py data_loader.py icv.py pages_denrees.py
```

Les fonds de carte peuvent nécessiter une connexion Internet. Les chiffres de la page Éducation indiquent leurs années de référence; certains indicateurs régionaux sont explicitement signalés « à confirmer ».
