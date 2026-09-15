# 07a — SentencePiece, option Broadcasting

- [Entraînement SentencePiece](sentencepiece_broadcast.ipynb#train-sp)
- [Alignement sous-tokens/tags](sentencepiece_broadcast.ipynb#alignment)
- [Entraînement du modèle](sentencepiece_broadcast.ipynb#training)
- [Évaluation au niveau mot](sentencepiece_broadcast.ipynb#evaluation)

Chaque sous-token reçoit le même tag que son mot d'origine. Le tagset ne change
pas (les 12 tags habituels). Évaluation reconvertie au niveau mot via
`reconstruct_word_tags`, pour rester comparable aux modules 05/06.

**Limite** : un mot découpé en plusieurs sous-tokens compte plusieurs fois le
même tag dans la loss, ce qui peut légèrement biaiser l'entraînement vers les
mots longs.