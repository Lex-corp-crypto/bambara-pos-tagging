# 09 — Diagnostic et comparaison de la tokenisation (v1 vs v2)

Suite directe du module 08 : une fois la performance "à froid" mesurée sur
CorBaMa, on identifie où la tokenisation v1 échoue, puis on compare avec la
v2 (déjà entraînée dans le module 07, guidée par les morphèmes réels de
CorBaMa) — pas de réentraînement ici, ce module recharge des checkpoints
existants.

## Contenu

- **`tokenization_diagnosis.ipynb`** : mots les plus fragmentés par
  SentencePiece v1 sur CorBaMa, et comparaison à la vraie décomposition
  morphologique fournie par les linguistes (balises `span.m`).
- **`retrain_and_compare.ipynb`** : recharge les modèles v1 et v2 déjà
  entraînés dans `07a_broadcast_scheme/` et `07b_bio_scheme/`, et les compare
  formellement sur notre propre test set (Bayelemabaga) et sur CorBaMa :
  accuracy, F1 par tag, fragmentation moyenne. Malgré son nom, ce notebook
  n'entraîne plus rien — le réentraînement de la v2 a déjà eu lieu au module 07.

## Prérequis

- Module 08 exécuté en entier (`corbama_corpus.tsv` doit exister).
- Module 07a exécuté en entier, v1 ET v2 (les checkpoints
  `bambara_pos_sp_broadcast.pth` et `bambara_pos_sp_broadcast_v2.pth` doivent
  exister).
- Module 07b exécuté en entier, v1 ET v2, si tu veux aussi la comparaison sur
  le schéma BIO (`bambara_pos_sp_bio.pth` et `bambara_pos_sp_bio_v2.pth`).

## Prochaine étape

Une fois que Valentin transmet plus de fichiers CorBaMa, ce texte pourra être
ajouté directement à l'entraînement d'une v3 du tokeniseur — en réservant
toujours une partie jamais utilisée en entraînement, pour garder un test
honnête.