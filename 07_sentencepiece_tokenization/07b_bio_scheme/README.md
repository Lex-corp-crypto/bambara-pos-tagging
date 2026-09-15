# 07b — SentencePiece, option BIO

- [Construction du tagset BIO](sentencepiece_bio.ipynb#bio-tagset)
- [Entraînement du modèle](sentencepiece_bio.ipynb#training)
- [Évaluation au niveau mot](sentencepiece_bio.ipynb#evaluation)

Réutilise le modèle SentencePiece entraîné dans `07a_broadcast_scheme/`. Le
premier sous-token d'un mot reçoit `B-TAG`, les suivants `I-TAG` — schéma BIO
standard. Tagset : 23 classes (`<PAD>` + B/I pour chacun des 11 tags).

**Prérequis** : exécuter `07a_broadcast_scheme` au moins une fois avant ce
notebook, pour que `bambara_sp.model` existe.

**Limite** : tagset deux fois plus grand pour un corpus qui reste modeste
(640k tokens), donc plus de paramètres à apprendre côté couche de sortie.