
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
plus fréquents.

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

Python, PyTorch, NumPy, Pandas, scikit-learn, Matplotlib.

## Architecture du projet

```text
.
├── bambara_pos_utils.py            Fonctions et classes partagées entre tous les notebooks
├── 01_data_preprocessing/          Tokenisation, vocabulaire
├── 02_model_architecture/          Modèle BiLSTM (LSTMTagger, LSTMTaggerWithBatch)
├── 03_training_and_batching/       Dataset, DataLoader, boucle d'entraînement de base
├── 04_bambara_pos_pipeline/        Chargement du corpus, lexique, re-étiquetage
├── 05_evaluation_and_diagnostics/  Métriques, class weights, diagnostic data leakage
├── 06_final_training_and_inference/ Entraînement final, hyperparamètres, inférence
└── README.md
```

Les notebooks se lisent dans l'ordre :

```text
01 Prétraitement → 02 Modèle → 03 Entraînement de base →
04 Corpus & Pipeline → 05 Évaluation & Diagnostic → 06 Entraînement final & Inférence
```

**Point d'architecture important** : chaque notebook est un kernel Jupyter
séparé. Les fonctions et classes communes (tokeniseur, modèle, Dataset,
lexique, chargement de checkpoint...) ne sont donc pas redéfinies dans chaque
fichier : elles vivent une seule fois dans `bambara_pos_utils.py`, importé en
haut de chaque notebook avec :

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from bambara_pos_utils import *
```

## Installation

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Utilisation

Pour reproduire l'entraînement complet, exécuter les notebooks dans l'ordre
01 → 06 (Restart Kernel + Run All pour chacun). Le notebook 06 produit le
modèle final dans `06_final_training_and_inference/models/bambara_pos_best.pth`.

Pour utiliser uniquement l'inférence, une fois le modèle entraîné :

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
référence.

### Tags les plus difficiles pour le modèle

Le notebook 06 identifie explicitement, à partir du F1-score par tag sur le
test set, quels tags le modèle confond le plus souvent. Sans surprise, les
tags les mieux couverts par le lexique (`AUX`, `PRON`, `POSTP`) obtiennent
des scores très élevés — ce qui est cohérent avec le point ci-dessus, pas
forcément une preuve de bonne généralisation. Les tags à faible effectif
(`ADV`, `PART`, `ADJ`) restent les plus fragiles, probablement par manque
d'exemples plutôt que par difficulté intrinsèque.

## Pipeline hybride

En inférence, le système combine trois niveaux :

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
- Comparer le BiLSTM à un modèle Transformer pré-entraîné (AfriBERTa,
  XLM-RoBERTa) sur les mêmes splits, une fois un jeu de test fiable disponible.
- Étendre la recherche d'hyperparamètres du notebook 06 (nombre de couches
  LSTM, dropout, taille d'embedding) si le temps de calcul le permet.
