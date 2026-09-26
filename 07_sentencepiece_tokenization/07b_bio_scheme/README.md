# 07b — SentencePiece, option BIO

- [Construction du tagset BIO](sentencepiece_bio.ipynb#bio-tagset)
- [Entraînement du modèle v1](sentencepiece_bio.ipynb#training)
- [Évaluation au niveau mot](sentencepiece_bio.ipynb#evaluation)
- [Mise à jour v2 (données CorBaMa)](sentencepiece_bio.ipynb#v2-corbama)

Réutilise le SentencePiece v1 de `07a`, puis compare avec le v2 (aussi entraîné
dans `07a`) sur le schéma BIO. Tagset : 22 classes (`<PAD>` + B/I pour chacun
des 10 tags hors PUNCT, PUNCT restant simple).

**Prérequis** : exécuter `07a_broadcast_scheme` en entier (v1 ET v2) avant ce
notebook, et le module 08 pour `corbama_corpus.tsv`.