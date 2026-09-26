# 07 — Tokenisation en sous-mots avec SentencePiece

À la demande du superviseur : remplacer la tokenisation mot-entier (modules 01-06)
par une tokenisation en sous-mots avec SentencePiece, pour réduire le problème des
mots inconnus sur une langue morphologiquement riche comme le bambara.

## Le problème d'alignement

Le corpus est annoté **au niveau du mot** (un mot = un tag). Découper un mot en
plusieurs sous-tokens casse cet alignement. Deux stratégies sont testées :

- **[07a_broadcast_scheme/](07a_broadcast_scheme/)** : chaque sous-token reçoit le
  même tag que le mot entier.
- **[07b_bio_scheme/](07b_bio_scheme/)** : `B-TAG` pour le premier sous-token,
  `I-TAG` pour les suivants.

## Deux versions du tokeniseur

- **v1** (`bambara_sp.model`) : entraîné sur Bayelemabaga seul, `vocab_size=4000`.
  C'est la version "avant" (module 07 initial).
- **v2** (`bambara_sp_v2.model`) : ajoutée suite au retour du superviseur après
  test sur les données réelles de Valentin (module 08). Toujours entraînée
  uniquement sur Bayelemabaga (**jamais sur CorBaMa**, pour ne pas fausser
  l'évaluation), mais avec `vocab_size=6000` et guidée par les vrais morphèmes
  bambara identifiés par les linguistes dans CorBaMa (`user_defined_symbols`).

Les deux versions sont comparées dans 07a et 07b, à la fois sur notre propre
test set (Bayelemabaga) et sur CorBaMa (module 08), avec l'accuracy et le taux
moyen de fragmentation (sous-tokens par mot) comme indicateurs.

## Limites à garder en tête

- Le broadcasting compte plusieurs fois le même tag dans la loss pour un mot
  multi-tokens.
- Le schéma BIO double la taille du tagset (22 classes au lieu de 11).
- Utiliser le vocabulaire de morphèmes de CorBaMa pour guider v2 est un choix
  intermédiaire : ça n'entraîne pas directement sur les phrases de CorBaMa (donc
  pas de fuite du signal d'évaluation), mais ça reste une information tirée de
  CorBaMa — à mentionner si on présente les résultats comme un test "à l'aveugle".
- Ce changement de tokenisation ne règle pas le problème de data leakage du
  corpus Bayelemabaga documenté au module 05 — il s'y ajoute.