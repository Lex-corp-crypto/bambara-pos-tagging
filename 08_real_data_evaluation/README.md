# 08 — Évaluation sur données réelles (CorBaMa)

À la demande du superviseur : tester les modèles déjà entraînés (modules 06/07)
sur de vraies données linguistiques, avant d'améliorer quoi que ce soit.

## Contenu

- **`corbama_parsing.ipynb`** : parse les fichiers `.dis.html` du sous-corpus
  désambiguïsé du Corpus Bambara de Référence (Vydrin, Maslinsky, Méric —
  CNRS/INALCO), génère automatiquement `corbama_corpus.tsv` (tagset mappé sur
  nos 11 classes) et `corbama_corpus_fine.tsv` (tagset fin, 24 classes).
- **`baseline_evaluation.ipynb`** : évalue les modèles des modules 06 et 07
  (v1), **sans aucun réentraînement**, sur ce corpus réel jamais vu.
- **`corbama-disamb/`** : fichiers HTML sources (à placer ici avant d'exécuter
  `corbama_parsing.ipynb`).

## Prérequis

- Le dossier `corbama-disamb/` doit être placé directement dans
  `08_real_data_evaluation/`, avec la même arborescence interne que fournie.
- `pip install beautifulsoup4`

## Points de vigilance sur le parsing

- La ponctuation (`span.c`) est traitée séparément des mots (`span.w`), dans
  l'ordre d'apparition — un premier essai l'ignorait complètement.
- Un artefact d'encodage (BOM, U+FEFF) en début de phrase est filtré — sinon
  il pollue le corpus comme fausse ponctuation.
- Le tagset fin de CorBaMa a 24 catégories au niveau mot (un premier comptage
  avait mélangé par erreur tags de mots et tags de morphèmes internes,
  donnant 27 à tort).

## Chiffres obtenus (sur les 50 fichiers fournis)

~4 640 phrases, ~65 600 tokens (mots + ponctuation). Distribution : NOM 22,1% ·
PUNCT 17,3% · PRON 14,0% · AUX 13,1% · VERBE 12,3% · POSTP 8,4% · DET 3,5% ·
CONJ 3,5% · PART 2,6% · ADJ 2,4% · ADV 0,8%.