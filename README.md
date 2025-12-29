# streamlit-ocr-translator

Application **Streamlit + Python** pour :
- ingérer un dossier `HTML + images`,
- découper le HTML en segments traçables,
- (optionnel) faire de l’OCR sur les images,
- extraire / gérer un glossaire,
- traduire par chunks (DeepL ou OpenAI) en **préservant la structure HTML**, en deux passes (MT propre puis révision académique),
- post-traiter + contrôler la qualité (rapport terminologique, contrôle des chiffres/citations),
- reconstruire un HTML final et exporter.

> ⚠️ Les appels aux APIs (OpenAI / DeepL) sont **prévu** dans le code, mais ils nécessitent vos clés.  
> Ce dépôt n’inclut aucune clé ni appel automatique.

---

## 1) Installation

### Option A — Local (venv)

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Option B — Docker

Voir `Dockerfile`. Exemple :

```bash
docker build -t streamlit-ocr-translator .
docker run --rm -it -p 8501:8501 -v "$PWD:/app" streamlit-ocr-translator
```

---

## 2) Configuration

Copiez les exemples :

```bash
cp .env.example .env
cp config.example.json config.json
```

- `.env` : mettez vos clés (`OPENAI_API_KEY` et/ou `DEEPL_AUTH_KEY`).
- `config.json` : choisissez le provider, les chemins, et les options (OCR, chunking, révision, QA).
- `style_guide.json` : guide de style machine-lisible (registre académique, conventions typographiques, collocations préférées) injecté dans tous les prompts.
- `style_guide.default.json` / `style_guide.schema.json` : base féodale/médiévale et schéma de validation strict ; le pipeline fusionne et refuse les guides incomplets avant exécution.
- `pipeline.resume` : laissez `true` pour réutiliser les artefacts déjà générés (segments, glossaire) et reprendre après un échec sans repartir de zéro.
- `translation.parallel_workers` / `revision.parallel_workers` : active un traitement multithread des chunks/segments pour accélérer le débit des appels API (adapter selon vos limites de quota).
- `translation.structure_guard` / `revision.structure_guard` : garde HTML stricte avec sentinelles + validation avant/après appel LLM.
- `translation.scheduling` / `revision.scheduling` : quotas (requêtes/min), nombre de retries et backoff exponentiel pour éviter le throttling.
- `translation.chunking` : découpage **contextuel** (sections + listes) pour préserver le fil logique ; ajustez `max_chars_per_chunk` / `max_segments_per_chunk` pour limiter la taille envoyée.
- `translation.translation_memory` : chemin de TM persistante (JSONL), seuil de fuzzy matching et tailles max pour réutiliser/reconcilier des traductions stables entre runs.
- `postproc.apply_glossary` + masquage en amont : les termes du glossaire sont normalisés (variantes/pluriels) et masqués avant appel LLM puis réinjectés après traduction/révision pour verrouiller la terminologie.

---

## 3) Données d’entrée

Placez vos fichiers dans :

```
data/raw/document.html
data/raw/images/
```

Vous pouvez aussi pointer `config.json` vers un autre dossier.

---

## 4) Exécuter en CLI

### A) Construire les segments
```bash
python scripts/build_segments.py --html data/raw/document.html --out data/segments/segments.json
```

### B) OCR (optionnel)
```bash
python scripts/ocr_images.py --images data/raw/images --out data/segments/images_ocr.json
```

### C) Traduire tout le document (pipeline)
```bash
python run_translate.py --config config.json
```

Cette commande produit désormais :
- `data/segments/term_candidates.json` : extraction multi-signal des candidats de glossaire (n-grams, typographie, patrons de définition, entités temporelles, cooccurrences).
- `data/segments/glossary_draft.json` et `data/segments/glossary_proposals.json` : glossaire décisionnel + propositions de traduction (ou placeholders si l'API n'est pas configurée).
- `data/translations/segments_translated.json` (pass 1) et `data/translations/segments_translated_v2.json` (pass 2 révisée).
- `data/qa/terminology_report.json` : déviations par rapport au glossaire.
- `data/qa/terminology_corrections.json` : patchs automatiques appliqués aux segments non conformes.
- `data/qa/register_report.json` : évaluation heuristique du registre (anglicismes/calques).
- `data/translations/translation_memory.jsonl` : mémoire de traduction persistante alimentée en continu (source, cible, HTML) pour réemploi et détection de variantes.
- `logs/audit.json` : trace consolidée des prompts hashés, collocations apprises et corrections appliquées.

### D) Reconstruire le HTML
```bash
python scripts/rebuild_html.py --html data/raw/document.html --segments data/translations/segments_translated.json --out data/exports/translated_document.html
```

---

## 5) Lancer l’UI Streamlit

```bash
streamlit run app.py
```

L’UI permet :
- de lancer les étapes,
- d’inspecter segments et traductions,
- d’éditer manuellement (post-édition),
- d’exporter le HTML final.

---

## 6) Notes importantes

- **Préservation HTML** : par défaut, le traducteur LLM attend/retourne un fragment HTML identique en structure.
  Le module `postproc.py` vérifie la structure : tags, attributs (`href`, `src`, `id`, `class`…), références images.
- **Glossaire** : extraction automatique + révision humaine recommandée (voir `src/glossary.py`).
- **Masquages de sécurité** : les termes du glossaire et les nombres/dates/citations sont masqués avant appel modèle puis réinjectés pour limiter les dérives.
- **Boucle QA** : rapport terminologique + corrections automatiques des segments non conformes, plus évaluation heuristique du registre académique.
- **OCR** : nécessite Tesseract installé sur votre machine (ou via Docker) + `pytesseract`.
- **QA** : le module `src/qa.py` propose une back-translation et des checks de cohérence.

---

## 7) Structure du projet

```
streamlit-ocr-translator/
├─ app.py
├─ run_translate.py
├─ scripts/
├─ src/
├─ data/
├─ logs/
└─ tests/
```

---

## Licence

MIT (voir fichier `LICENSE` si vous en ajoutez un).
