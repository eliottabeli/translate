# Audit des correctifs appliqués

## Bugs techniques résolus
- `chunk_segments_contextual` renvoie désormais un itérateur vide lorsque la liste de segments est vide, évitant les `TypeError` quand la mémoire de traduction couvre 100 % du document.【F:src/utils.py†L96-L140】
- L’étape QA respecte le flag `qa.enabled` du fichier de configuration : le pipeline saute proprement la QA lorsque désactivée.【F:run_translate.py†L310-L337】
- Les traducteurs OpenAI/DeepL signalent l’absence de clé API avec un message clair et l’interface Streamlit relaie l’erreur sans crash brutal.【F:src/translator.py†L84-L116】【F:app.py†L82-L137】【F:run_translate.py†L214-L252】
