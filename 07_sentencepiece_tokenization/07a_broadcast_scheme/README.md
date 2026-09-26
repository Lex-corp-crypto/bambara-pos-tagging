# 07a — SentencePiece, option Broadcasting

- [Entraînement SentencePiece v1](sentencepiece_broadcast.ipynb#train-sp)
- [Alignement sous-tokens/tags](sentencepiece_broadcast.ipynb#alignment)
- [Entraînement du modèle v1](sentencepiece_broadcast.ipynb#training)
- [Évaluation au niveau mot](sentencepiece_broadcast.ipynb#evaluation)
- [Mise à jour v2 (données CorBaMa)](sentencepiece_broadcast.ipynb#v2-corbama)
- [Comparaison v1 vs v2](sentencepiece_broadcast.ipynb#v1-vs-v2)

Chaque sous-token reçoit le même tag que son mot d'origine. La v2 est guidée par
les morphèmes réels de CorBaMa sans être entraînée sur ses phrases, pour garder
un test honnête sur données jamais vues (module 08).

**Prérequis** : avoir exécuté le module 08 (`corbama_corpus.tsv` doit exister)
avant la section v2 de ce notebook.