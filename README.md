# streamlit-ocr-translator

Application **Streamlit + Python** pour :
- ingérer un dossier `HTML + images`,
- découper le HTML en segments traçables,
- (optionnel) faire de l’OCR sur les images,
- extraire / gérer un glossaire,
- traduire par chunks (DeepL ou OpenAI) en **préservant la structure HTML**,
- post-traiter + contrôler la qualité,
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
- `config.json` : choisissez le provider, les chemins, et les options (OCR, chunking, QA).

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
