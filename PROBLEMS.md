# Audit des problèmes actuels

## Exécution et tests
- `pytest` échoue dès la collecte : les tests ne peuvent pas importer le paquet `src` (erreur `ModuleNotFoundError`), ce qui rend toute la suite de tests inutilisable en l'état. 【636a94†L1-L33】

## Problèmes de structure de fichier / syntaxe
- Plusieurs fichiers commencent par un caractère d'échappement `\` isolé, ce qui rend les fichiers invalides lorsqu'ils sont exécutés directement depuis leurs propres répertoires ou par certains outils :
  - `src/html_parser.py` 【F:src/html_parser.py†L1-L4】
  - `src/utils.py` 【F:src/utils.py†L1-L4】
  - `app.py` 【F:app.py†L1-L6】
  - `src/qa.py` 【F:src/qa.py†L1-L6】
  - `tests/test_html_parser.py` 【F:tests/test_html_parser.py†L1-L3】
  - `tests/test_postproc.py` 【F:tests/test_postproc.py†L1-L3】
  - `tests/test_translator.py` 【F:tests/test_translator.py†L1-L4】
  - `scripts/build_segments.py` 【F:scripts/build_segments.py†L1-L6】

## Problèmes d'importations et de dépendances
- Les tests de base n'initialisent pas le chemin d'import (`PYTHONPATH`) pour inclure la racine du projet, ce qui provoque l'erreur `ModuleNotFoundError` observée. Aucun `conftest.py` ni installation du paquet n'est présent pour corriger le chemin. 【636a94†L1-L33】
- Plusieurs modules déclarent des imports inutilisés qui compliquent la lecture et signalent un manque de nettoyage :
  - `src/html_parser.py` importe `dataclass` sans l'utiliser. 【F:src/html_parser.py†L4-L9】
  - `src/qa.py` importe `json` et `Tuple` sans usage. 【F:src/qa.py†L4-L12】
  - `run_translate.py` importe `BeautifulSoup` mais ne l'utilise pas. 【F:run_translate.py†L7-L13】

## Couverture fonctionnelle et robustesse
- Aucune validation d'entrée ni gestion d'erreur n'entoure les appels Streamlit/CLI pour les chemins fournis par l'utilisateur, ce qui peut provoquer des crashs immédiats si un fichier manquant est référencé. Exemple : `storage.read_json` est appelé sans vérification d'existence dans `app.py` lors du chargement initial. 【F:app.py†L33-L57】【F:app.py†L60-L70】
- Les commandes CLI (`scripts/*.py`) ne gèrent pas les exceptions d'E/S ou de parsing et s'arrêtent brutalement en cas de problème (fichier absent, HTML invalide), sans message utilisateur adapté. 【F:scripts/build_segments.py†L9-L24】【F:scripts/rebuild_html.py†L9-L24】

## Qualité du code et maintenabilité
- Les fichiers tests et modules clés débutent par un backslash superflu, signe probable de résidus d'édition ou de copier-coller non nettoyés (voir section Syntaxe). Cela suggère un manque de revue et empêchera l'analyse statique ou la distribution du paquet via `pyproject`/`setup.py`.
- Le dépôt ne contient ni configuration d'environnement (ex. `pyproject.toml`, `setup.cfg`) ni scripts d'installation, alors que les modules sont importés comme un paquet (`src.*`). Cela rend l'installation et l'exécution dépendantes du répertoire courant et casse dès que les tests sont lancés depuis un autre répertoire. 【636a94†L1-L33】

