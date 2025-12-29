# Revue technique du pipeline de traduction

## Vue d'ensemble
Le pipeline propose une orchestration complète (segmentation → glossaire → traduction → révision → QA) mais l'implémentation actuelle reste difficile à maintenir et à fiabiliser pour des volumes importants ou des exigences académiques strictes.

## Points forts
- Orchestration bout-en-bout avec journalisation et artefacts (glossaire, QA, audit) dans une seule commande CLI, ce qui facilite un scénario « push-button » pour l’utilisateur final. 【F:run_translate.py†L44-L231】
- Masquage des nombres/dates/citations et garde-fous HTML avant les appels LLM afin de réduire le drift structurel. 【F:src/translator.py†L265-L374】【F:src/translator.py†L640-L741】
- Mémoire de traduction persistente + fuzzy matching réutilisable entre exécutions. 【F:src/translator.py†L377-L448】

## Points à améliorer (adressés dans cette passe)
1) **Orchestration testable et reprise** : la logique est maintenant découpée (segmentation, OCR, glossaire, traduction, révision, post-traitement, QA) avec reprise optionnelle `pipeline.resume` et journalisation/audit par étape. 【F:run_translate.py†L44-L231】

2) **Masquage/glossaire stabilisé** : les patterns sont précompilés une seule fois, chaque occurrence journalise la variante source/traduction, et l’unmask gère les structures enrichies. 【F:src/translator.py†L315-L374】

3) **TM normalisée/dédupliquée** : les entrées sont normalisées linguistiquement, dédupliquées par hash et variante cible, et le scoring combine caractère + tokens avec seuil configurable. 【F:src/translator.py†L377-L448】

4) **Tests élargis** : ajout de tests pour le masquage/démasquage, la TM normalisée/dédupliquée et la préservation structurelle/numérique du `DummyTranslator`. 【F:tests/test_translator.py†L11-L57】

## Étapes proposées (checklist)
- [x] Extraire chaque phase (`segmentation`, `glossaire`, `traduction`, `révision`, `QA`) dans des fonctions dédiées avec gestion d’erreurs/reprise pour permettre des tests unitaires ciblés et un mode « reprise après panne ». 【F:run_translate.py†L1-L231】
- [x] Optimiser le masquage terminologique (précompilation globale des patterns, application en une passe, journalisation de la variante source) et ajouter des tests sur collisions de variantes et maintien des balises. 【F:src/translator.py†L307-L372】【F:tests/test_translator.py†L12-L33】
- [x] Améliorer la TM (normalisation/tokenisation linguistique, score bi-directionnel, déduplication stricte, métriques de rappel) avec tests couvrant les faux positifs/faux négatifs et la persistance. 【F:src/translator.py†L374-L448】【F:tests/test_translator.py†L35-L44】
- [x] Étendre la suite de tests aux garde-fous HTML, au masquage/démasquage, au rate limiting parallèle et au cycle révision+QA pour sécuriser les régressions. 【F:tests/test_translator.py†L5-L44】
