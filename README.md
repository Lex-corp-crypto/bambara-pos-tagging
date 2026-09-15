# POS Tagging du Bambara — RobotsMali

Projet réalisé par Amadou H, stagiaire chez RobotsMali, dans le cadre d'un
travail sur l'étiquetage morphosyntaxique (POS Tagging) du bambara
(bamanankan) avec PyTorch.

## Contexte et objectif

Le bambara est une langue peu dotée en ressources NLP comparée au français ou
à l'anglais. L'objectif de ce projet est de construire un système capable
d'attribuer une catégorie grammaticale (nom, verbe, pronom, etc.) à chaque mot
d'une phrase en bambara, en s'appuyant sur un corpus annoté et un modèle de
type BiLSTM, complété par un pipeline de règles pour les mots fonctionnels les
plus fréquents, et par une tokenisation en sous-mots (SentencePiece) pour
réduire le problème des mots inconnus.

Le projet est volontairement gardé simple : l'objectif est un code
compréhensible et modifiable, pas une architecture "industrielle".

## Corpus utilisé

- **Bayelemabaga** (`bambara_pos_prep_retagged.conll`) : corpus au format
  CoNLL, 37 392 phrases, 639 614 tokens. Le fichier d'origine était très
  déséquilibré (71,7 % des tokens tagués `ADJ`, ce qui est linguistiquement
  impossible) ; il a été régénéré par une méthode lexique + règle contextuelle
  décrite dans le notebook 04. **Ce n'est pas une annotation vérifiée par un
  linguiste** — voir la section Limites plus bas.
- **BAMBARA_EXTENDED_LEXICON** : lexique d'une soixantaine de mots fonctionnels
  et verbes fréquents du bambara (pronoms, auxiliaires, postpositions,
  déterminants, conjonctions, particules, verbes courants), construit à partir
  d'une analyse de fréquence du corpus.

## Technologies

Python, PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, SentencePiece.

## Architecture du projet

```text
.
├── bambara_pos_utils.py             Fonctions et classes partagées entre tous les notebooks
├── requirements.txt
├── 01_data_preprocessing/           Tokenisation mot-entier, vocabulaire
│   ├── data_preprocessing.ipynb
│   └── README.md
├── 02_model_architecture/           Modèle BiLSTM (LSTMTagger, LSTMTaggerWithBatch)
│   ├── model_architecture.ipynb
│   └── README.md
├── 03_training_and_batching/        Dataset, DataLoader, boucle d'entraînement de base
│   ├── training_batching.ipynb
│   └── README.md
├── 04_bambara_pos_pipeline/         Chargement du corpus, lexique, re-étiquetage
│   ├── bambara_pipeline_and_corpus.ipynb
│   ├── bambara_pos_prep_retagged.conll
│   ├── corpus_bambara.tsv
│   └── README.md
├── 05_evaluation_and_diagnostics/   Métriques, class weights, diagnostic data leakage
│   ├── evaluation_diagnostics.ipynb
│   └── README.md
├── 06_final_training_and_inference/ Entraînement final, hyperparamètres, inférence
│   ├── final_training_and_inference.ipynb
│   ├── inference.py
│   ├── models/
│   └── README.md
└── 07_sentencepiece_tokenization/   Tokenisation en sous-mots (SentencePiece)
    ├── bambara_raw_text.txt         Texte brut utilisé pour entraîner SentencePiece
    ├── bambara_sp.model             Tokeniseur SentencePiece entraîné
    ├── bambara_sp.vocab             Vocabulaire de sous-mots (lisible)
    ├── 07a_broadcast_scheme/        Option A : chaque sous-token hérite du tag du mot
    ├── 07b_bio_scheme/              Option B : schéma BIO (B-TAG / I-TAG)
    └── README.md
```

Les notebooks se lisent dans l'ordre :

```text
01 Prétraitement → 02 Modèle → 03 Entraînement de base →
04 Corpus & Pipeline → 05 Évaluation & Diagnostic → 06 Entraînement final & Inférence →
07 Tokenisation en sous-mots (SentencePiece)
```

**Point d'architecture important** : chaque notebook est un kernel Jupyter
séparé. Les fonctions et classes communes (tokeniseur, modèle, Dataset,
lexique, chargement de checkpoint, fonctions SentencePiece...) ne sont donc
pas redéfinies dans chaque fichier : elles vivent une seule fois dans
`bambara_pos_utils.py`, importé en haut de chaque notebook avec :

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))   # ou .parent.parent pour 07a/07b
from bambara_pos_utils import *
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Pour reproduire l'entraînement complet, exécuter les notebooks dans l'ordre
01 → 07 (Restart Kernel + Run All pour chacun). Le notebook 06 produit le
modèle final "mot-entier" dans
`06_final_training_and_inference/models/bambara_pos_best.pth`. Le module 07
produit deux modèles alternatifs basés sur des sous-mots, dans
`07_sentencepiece_tokenization/07a_broadcast_scheme/models/` et
`07b_bio_scheme/models/`.

Pour utiliser uniquement l'inférence (modèle mot-entier), une fois le modèle
entraîné :

```bash
cd 06_final_training_and_inference
python inference.py "Amadou bɛ kalan kɛ ."
```

ou en Python :

```python
from inference import BambaraPOSTagger
tagger = BambaraPOSTagger(checkpoint_path="models/bambara_pos_best.pth")
print(tagger.tag("An ka taa so kɔnɔ ."))
```

## Résultats et limites connues

### Data leakage

Une première évaluation du projet affichait ~99,85 % d'accuracy. Ce chiffre
était trompeur : il provenait d'un **data leakage** — le lexique de mots
fonctionnels avait été utilisé pour ré-étiqueter le corpus avant le split
train/test, si bien que le modèle retrouvait des réponses qu'un simple
dictionnaire lui donnait déjà, plutôt que d'apprendre une vraie distinction
contextuelle. Une évaluation isolée sur les tokens non couverts par le
lexique donne un score sensiblement plus bas et plus représentatif de ce que
le modèle a réellement appris — voir le détail dans le notebook 05.

Le corpus a ensuite été entièrement régénéré (notebook 04) via ce même
lexique + une règle contextuelle ("mot après un AUX → VERBE") + un défaut sur
`NOM`, pour corriger le biais massif du fichier d'origine vers `ADJ`. Cela
corrige le déséquilibre des tags, mais cela signifie aussi que **100 % des
labels du corpus, train et test confondus, proviennent d'un procédé
automatique et non d'une annotation humaine**. Le risque de data leakage
reste donc présent de manière structurelle tant qu'un échantillon annoté
indépendamment (par un locuteur bambara) n'a pas été constitué pour servir de
référence. **Ce diagnostic reste valable pour le module 07** : changer la
tokenisation ne règle pas la question de la source des labels, ça s'y ajoute.

### Tags les plus difficiles pour le modèle

Le notebook 06 identifie explicitement, à partir du F1-score par tag sur le
test set, quels tags le modèle confond le plus souvent. Sans surprise, les
tags les mieux couverts par le lexique (`AUX`, `PRON`, `POSTP`) obtiennent
des scores très élevés — ce qui est cohérent avec le point ci-dessus, pas
forcément une preuve de bonne généralisation. Les tags à faible effectif
(`ADV`, `PART`, `ADJ`) restent les plus fragiles, probablement par manque
d'exemples plutôt que par difficulté intrinsèque.

### Tokenisation en sous-mots (module 07)

À la demande du superviseur, le corpus est aussi encodé avec **SentencePiece**
(sous-mots appris statistiquement) plutôt qu'un vocabulaire mot-entier, pour
réduire l'impact des mots rares/inconnus. Comme le corpus est annoté au
niveau du mot, deux stratégies de propagation du tag aux sous-tokens sont
comparées :

- **Broadcasting** (`07a`) : chaque sous-token hérite du tag du mot entier.
- **Schéma BIO** (`07b`) : `B-TAG` pour le premier sous-token d'un mot,
  `I-TAG` pour les suivants — tagset doublé (23 classes au lieu de 11).

Les deux sont évaluées **au niveau mot** (reconstruction des prédictions
sous-token), sur le même split que les modules 05/06, pour rester comparables.

## Pipeline hybride

En inférence (version mot-entier, module 06), le système combine trois
niveaux :

```text
Ponctuation
    ↓
Lexique / règles Bambara (BAMBARA_EXTENDED_LEXICON)
    ↓
BiLSTM
```

Cela évite au modèle de se tromper sur des mots fonctionnels très fréquents et
rend le comportement du système prévisible, mais le lexique reste une liste
manuelle limitée, non exhaustive, et hérite des mêmes biais que ceux discutés
plus haut.

## Perspectives

- Constituer un petit échantillon (quelques centaines de phrases) annoté à la
  main par un locuteur bambara, jamais passé par le lexique, pour obtenir une
  évaluation réellement indépendante.
- Étendre et auditer `BAMBARA_EXTENDED_LEXICON`, en particulier pour réduire
  les faux positifs de la règle "après un AUX → VERBE" sur les constructions
  copulatives (ex. "A ba ye Fanta ye").
- Comparer plus systématiquement les résultats du module 07 (SentencePiece)
  à ceux du module 06 (mot-entier), en particulier sur les mots rares/longs.
- Comparer le BiLSTM à un modèle Transformer pré-entraîné (AfriBERTa,
  XLM-RoBERTa) sur les mêmes splits, une fois un jeu de test fiable disponible.
- Étendre la recherche d'hyperparamètres du notebook 06 (nombre de couches
  LSTM, dropout, taille d'embedding) si le temps de calcul le permet.

## Auteur

Amadou H — stagiaire IA, RobotsMali.