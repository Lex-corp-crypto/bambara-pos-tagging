# 05 — Évaluation & Diagnostic

- [Métriques](evaluation_diagnostics.ipynb#metrics)
- [Class weights](evaluation_diagnostics.ipynb#class-weights)

Tagset : ADJ, ADV, AUX, CONJ, DET, NOM, PART, POSTP, PRON, PUNCT, VERBE, `<PAD>`.

Contenu : classification_report, matrice de confusion, analyse des confusions
les plus fréquentes, pondération des classes rares (`compute_class_weight`).

## Point critique : data leakage

Le corpus entier (notebook 04) a été re-étiqueté par `BAMBARA_EXTENDED_LEXICON`
et une règle contextuelle, pas par une annotation humaine. Toute accuracy
mesurée sur ce corpus surestime donc la performance réelle du modèle, puisque
les labels de test proviennent de la même source déterministe que celle
utilisée en inférence (pipeline hybride). Le notebook isole une accuracy
"hors lexique" pour donner une estimation plus réaliste, et documente
pourquoi seule une annotation humaine indépendante donnerait une mesure
totalement fiable.
