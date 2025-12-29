# Pipeline critique et plan d'amélioration

## Analyse critique (sans complaisance)
- La préservation de structure HTML repose uniquement sur la discipline du modèle via les prompts (cf. pass 1 et 2), sans garde-fous systématiques avant l'appel : aucun nettoyage ni neutralisation des attributs sensibles, ce qui laisse la porte ouverte à des corruptions si le modèle dérive ou si des balises exotiques apparaissent.
- La parallélisation introduite dans `translate_segments`/`revise_segments` manque de contrôle d'exécution : absence de backpressure, aucune stratégie de retry/exponential backoff ni budget de tokens par minute, donc risque élevé de throttling API et d'incohérences si des threads échouent silencieusement.
- Le traitement « chunké » reste naïf : on découpe par taille (`max_chars_per_chunk`/`max_segments_per_chunk`) sans tenir compte des frontières logiques (sections, listes, tableaux). Les segments consécutifs peuvent être séparés, ce qui casse les anaphores et le contexte argumentatif, surtout pour un manuel historique.
- La mémoire de traduction est purement append-only et locale au run : pas de persistance, pas de fuzzy matching, et aucune détection d'incohérence entre les variantes de traduction collectées, donc peu d'impact réel sur la stabilité terminologique.
- Le style guide est injecté mais non validé : aucune vérification de complétude (registre, conventions, préférences), pas de fallback structuré ni de merge avec un guide par défaut contextualisé pour l'histoire médiévale, ce qui réduit l'effet sur le registre académique.
- Le glossaire forcé n'est pas appliqué en amont des appels LLM : on compte sur le modèle pour le respecter, puis sur un post-traitement simple qui ne gère pas les flexions, la casse, ni les variantes de multi-mots, laissant persister des divergences.
- La QA terminologique (`terminology_report`) est déclenchée après coup mais sans seuils de sévérité ni blocage pipeline : aucune boucle de correction automatique ou semi-automatique pour réinjecter les écarts détectés.
- Pas de contrôle systématique des nombres/dates avant envoi : seule la vérification après traduction est faite, donc le modèle peut déjà modifier des chiffres sans garde-fou fort (pas de templating / masquage en entrée).
- Journalisation lacunaire : pas de trace des prompts (hashés ou tronqués) ni des décisions de glossaire par terme, ce qui nuit à la traçabilité exigée pour un flux académique.

## Plan d'amélioration (étapes actionnables)
- [x] **Robustifier la conservation HTML** : ajouter un pré-traitement qui enveloppe les segments avec des marqueurs sentinelles, insère un validateur HTML strict avant/après appel LLM, et refuse/retente si la structure diverge.
- [x] **Pilotage de parallélisation et quotas** : introduire une couche de scheduler avec files de tâches, retries exponentiels, limites de requêtes par minute et journalisation des erreurs par segment ; exposer ces paramètres dans la config.
- [x] **Chunking contextuel** : regrouper les segments par sections logiques (titres + paragraphes, listes complètes, cellules de tableau) via les métadonnées d'extraction, pour maintenir la cohérence discursive et réduire les ruptures d'anaphores.
- [x] **Mémoire de traduction persistance + fuzzy matching** : stocker la TM sur disque (SQLite/JSONL), ajouter une recherche approximative (distance éditoriale/embeddings locaux) et détecter les variantes FR d'un même EN pour verrouiller le glossaire.
- [x] **Validation et enrichissement du style guide** : définir un schéma JSON avec validation stricte, fusionner avec un guide par défaut médiéval/feodal, et intégrer un check de complétude avant lancement du pipeline.
- [x] **Application linguistique du glossaire en amont** : normaliser les candidats (lemmatisation, casse), générer des variantes fléchies, appliquer une substitution contrôlée avant l'appel LLM (mask/unmask) pour éviter les écarts, puis réappliquer en post-prod.
- [ ] **Boucle QA corrective** : si `terminology_report` détecte des écarts, auto-générer un patch (substitutions guidées) ou déclencher une micro-révision ciblée des segments fautifs avant export.
- [x] **Boucle QA corrective** : si `terminology_report` détecte des écarts, auto-générer un patch (substitutions guidées) ou déclencher une micro-révision ciblée des segments fautifs avant export.
- [x] **Masquage des nombres/dates/citations** : introduire un pré-processing qui remplace les nombres, dates et références par des jetons protégés, vérifie le compte avant/après, puis restaure, pour verrouiller les éléments factuels.
- [x] **Traçabilité renforcée** : loguer les prompts (hash + échantillon), les décisions glossaire (terme, choix FR, justification), les collocations apprises, et produire un rapport d'audit consolidé par run.
- [x] **Évaluation automatique de registre** : ajouter un module qui note la sortie (perplexité par modèle FR, détection d'anglicismes/calcques fréquents) et alerte si le niveau « académique » n'est pas atteint.
