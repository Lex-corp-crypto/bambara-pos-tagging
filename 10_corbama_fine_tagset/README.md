# 10 — Exploration du tagset fin de CorBaMa (24 catégories)

Le module 10 entraîne directement sur CorBaMa, avec son tagset fin détecté
automatiquement (24 catégories linguistiques + `PUNCT` + `<PAD>` = 26 classes).

## Différence avec les modules 08/09

Les modules 08 et 09 utilisaient CorBaMa comme **jeu de test externe** pour des
modèles entraînés sur Bayelemabaga (tagset réduit à 11 classes). Ici, on
entraîne **directement sur CorBaMa**, avec son tagset fin détecté
automatiquement (24 catégories linguistiques + `PUNCT` + `<PAD>` = 26 classes).

## Contenu

- **`word_tokenizer_experiment.ipynb`** : split CorBaMa (`seed=42`), tagset fin,
  entraînement mot-entier, évaluation.
- **`sentencepiece_experiment.ipynb`** : même split, tokeniseur SentencePiece
  entraîné uniquement sur le train split de CorBaMa (`vocab_size=1500`, plus
  petit que celui de Bayelemabaga, corpus plus modeste), puis modèle sous-mots
  (broadcasting), évaluation.
- **`comparison.ipynb`** : recharge les deux modèles déjà entraînés et compare
  accuracy, F1 par tag et taux de mots inconnus, sur le même test set.

## Limite à assumer clairement

CorBaMa (3711 phrases d'entraînement dans ce split) est bien plus petit que
Bayelemabaga (37 392 phrases). Les résultats de ce module sont une
**exploration**, pas une validation définitive du tagset fin — en particulier
les catégories rares (`onomat`, `intj`, `n/v`...) auront probablement trop peu
d'exemples pour être bien apprises. Plus de fichiers de Valentin amélioreraient
directement cette expérience.