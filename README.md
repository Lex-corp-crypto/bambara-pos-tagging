# POS Tagging du Bambara — RobotsMali

Projet réalisé par Amadou H, stagiaire chez RobotsMali, dans le cadre d'un
travail sur l'étiquetage morphosyntaxique (POS Tagging) du bambara
(bamanankan) avec PyTorch.

## Contexte et objectif

Le bambara est une langue peu dotée en ressources NLP comparée au français ou
à l'anglais. L'objectif de ce projet est de construire un système capable
d'attribuer une catégorie grammaticale à chaque mot d'une phrase en bambara,
en s'appuyant d'abord sur un corpus semi-automatique (Bayelemabaga) et un
modèle BiLSTM, complété par un pipeline de règles et une tokenisation en
sous-mots (SentencePiece), puis validé et affiné sur des données linguistiques
réelles (CorBaMa).

Le projet est volontairement gardé simple : l'objectif est un code
compréhensible et modifiable, pas une architecture "industrielle".

## Corpus utilisés

- **Bayelemabaga** (`bambara_pos_prep_retagged.conll`) : corpus principal
  d'entraînement, format CoNLL, 37 392 phrases, 639 614 tokens. Le fichier
  d'origine était très déséquilibré (71,7% des tokens tagués `ADJ`) ; il a été
  régénéré par lexique + règle contextuelle (module 04). **Ce n'est pas une
  annotation vérifiée par un linguiste.**
- **CorBaMa** (`corbama_corpus.tsv` / `corbama_corpus_fine.tsv`, module 08) :
  extrait du sous-corpus désambiguïsé du **Corpus Bambara de Référence**
  (Vydrin, Maslinsky, Méric — CNRS/INALCO), annotation linguistique réelle.
  50 fichiers, ~4 640 phrases, ~65 600 tokens. Utilisé comme jeu de test
  indépendant (modules 08/09) ET comme corpus d'entraînement à part entière
  pour l'exploration du tagset fin (module 10).
- **BAMBARA_EXTENDED_LEXICON** : lexique de mots fonctionnels et verbes
  fréquents, initialement construit par heuristique de fréquence, puis
  **révisé à partir du vocabulaire confirmé par CorBaMa** (voir plus bas).

## Technologies

Python, PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, SentencePiece,
BeautifulSoup4.

## Architecture du projet

```text
.
├── bambara_pos_utils.py             Fonctions et classes partagées entre tous les notebooks
├── requirements.txt
├── 01_data_preprocessing/           Tokenisation mot-entier, vocabulaire
├── 02_model_architecture/           Modèle BiLSTM (LSTMTagger, LSTMTaggerWithBatch)
├── 03_training_and_batching/        Dataset, DataLoader, boucle d'entraînement de base
├── 04_bambara_pos_pipeline/         Chargement du corpus, lexique, re-étiquetage
├── 05_evaluation_and_diagnostics/   Métriques, class weights, diagnostic data leakage
├── 06_final_training_and_inference/ Entraînement final, hyperparamètres, inférence
│   ├── final_training_and_inference.ipynb
│   ├── inference.py
│   ├── models/
│   └── README.md
├── 07_sentencepiece_tokenization/   Tokenisation en sous-mots — v1 ET v2
│   ├── bambara_raw_text.txt
│   ├── bambara_sp.model             v1 : entraîné sur Bayelemabaga seul
│   ├── bambara_sp_v2.model          v2 : vocab élargi, guidé par les morphèmes de CorBaMa
│   ├── 07a_broadcast_scheme/
│   ├── 07b_bio_scheme/
│   └── README.md
├── 08_real_data_evaluation/         Test "à froid" sur données réelles (CorBaMa)
│   ├── corbama_parsing.ipynb
│   ├── baseline_evaluation.ipynb
│   ├── corbama-disamb/              Fichiers HTML sources
│   ├── corbama_corpus.tsv           Corpus converti (tagset réduit, 11 classes)
│   ├── corbama_corpus_fine.tsv      Corpus converti (tagset fin, 24+ classes)
│   └── README.md
├── 09_improved_tokenization/        Diagnostic + comparaison v1/v2 de la tokenisation
│   ├── tokenization_diagnosis.ipynb
│   ├── retrain_and_compare.ipynb
│   └── README.md
└── 10_corbama_fine_tagset/          Entraînement direct sur CorBaMa, tagset fin
    ├── word_tokenizer_experiment.ipynb
    ├── sentencepiece_experiment.ipynb
    ├── comparison.ipynb
    ├── models/
    └── README.md
```

Les notebooks se lisent dans l'ordre :

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 (tokenisation v1 + v2) →
08 (test sur données réelles) → 09 (diagnostic + comparaison v1/v2) →
10 (entraînement direct sur CorBaMa, tagset fin)
```

**Point d'architecture important** : chaque notebook est un kernel Jupyter
séparé. Le code commun vit une seule fois dans `bambara_pos_utils.py`, importé
en haut de chaque notebook :

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))          # ou .parent.parent pour 07a/07b
from bambara_pos_utils import *
```

**Deux fonctions de chargement de checkpoint à ne pas confondre** :
- `load_checkpoint` : pour les modèles entraînés par l'ancienne boucle du
  module 06 (mot-entier, tagset fixe à 11 classes, checkpoint contenant
  `word_to_ix`).
- `load_checkpoint_generic` : pour les modèles entraînés par la `run_training`
  générique (modules 07, 09, 10), dont le checkpoint contient `vocab_size` et
  `tagset_size` mais pas de vocabulaire — à reconstruire séparément
  (`build_vocab` pour un modèle mot-entier, ou recharger le `.model`
  SentencePiece pour un modèle sous-mots).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Reproduire l'entraînement complet : exécuter les notebooks dans l'ordre 01 →
10 (Restart Kernel + Run All pour chacun — important après toute modification
de `bambara_pos_utils.py`, sinon le kernel garde l'ancienne version en
mémoire). Inférence (modèle mot-entier, module 06) :
```bash
cd 06_final_training_and_inference
python inference.py "Amadou bɛ kalan kɛ ."
```

## Résultats et limites connues

### Data leakage (corpus Bayelemabaga)

Une première évaluation affichait ~99,85% d'accuracy, un chiffre trompeur dû à
un data leakage : le lexique de mots fonctionnels avait servi à générer les
labels de tout le corpus, train et test confondus, avant le split. Le corpus
`bambara_pos_prep_retagged.conll` reste entièrement étiqueté par un procédé
automatique, pas par un linguiste — diagnostic complet au module 05.

### Validation sur données réelles (modules 08/09)

Les modèles ont été évalués sur **CorBaMa**, annoté par des linguistes et
indépendant du lexique et du pipeline de Bayelemabaga — le test le plus
honnête réalisé sur ce projet. Le module 09 diagnostique où la tokenisation
SentencePiece v1 fragmente mal les mots (comparaison à la vraie morphologie de
CorBaMa) et compare v1 vs v2.

### Tokenisation en sous-mots v1 et v2 (module 07)

Deux stratégies d'alignement tag/sous-token : broadcasting (07a) et schéma BIO
(07b). La v2 du tokeniseur SentencePiece (vocab élargi à 6000, guidé par les
vrais morphèmes bambara de CorBaMa via `user_defined_symbols`) reste entraînée
**uniquement sur le texte de Bayelemabaga** — jamais sur les phrases de
CorBaMa — pour que la comparaison sur CorBaMa reste un test indépendant.

### Lexique étendu — mise à jour suite à CorBaMa

`BAMBARA_EXTENDED_LEXICON` a été révisé à partir du vocabulaire fonctionnel
confirmé par CorBaMa (catégories fiables `pm`, `cop`, `pp`, `dtm`) : **46 mots
non ambigus ajoutés** (pureté ≥ 90% dans une seule catégorie), et **8 entrées
existantes corrigées** suite à un désaccord avec CorBaMa — notamment `ye`,
l'un des mots les plus fréquents du bambara, qui s'avère être majoritairement
une **postposition** (1300 occurrences) et non un auxiliaire (1065
occurrences pm+cop) comme on le supposait initialement par heuristique de
fréquence. `CORBAMA_AMBIGUOUS_EXCLUDED` documente 18 mots volontairement
exclus du lexique car réellement ambigus selon le contexte (ex. `bi` : 70%
`AUX`, mais aussi `NOM`/`ADV`/`ADJ`).

**Important** : cette correction change le comportement du lexique et du
pipeline hybride **à partir de maintenant**, mais ne régénère pas
automatiquement `bambara_pos_prep_retagged.conll`, qui reste tel qu'il a été
généré avec l'ancienne version du lexique.

### Exploration du tagset fin de CorBaMa (module 10)

Le module 10 entraîne directement sur CorBaMa
(pas Bayelemabaga), avec son tagset fin détecté automatiquement (24
catégories linguistiques + `PUNCT` + `<PAD>` = 26 classes), en comparant
tokenisation mot-entier et SentencePiece sur un même train/test split
(`seed=42`, 3711/463/465 phrases). **Résultat à lire comme une exploration** :
CorBaMa reste petit comparé à Bayelemabaga, donc les catégories rares du
tagset fin (`onomat`, `intj`, `n/v`...) manquent probablement de données pour
être bien apprises.

## Pipeline hybride

En inférence (modèle mot-entier, module 06) :
```text
Ponctuation → Lexique / règles Bambara (BAMBARA_EXTENDED_LEXICON) → BiLSTM
```

## Perspectives

- Intégrer plus de fichiers CorBaMa dès que Valentin les transmet, en
  réservant toujours une partie jamais utilisée en entraînement.
- Régénérer `bambara_pos_prep_retagged.conll` avec le lexique corrigé, pour
  propager les 8 corrections (dont `ye`) jusqu'au corpus d'entraînement
  principal.
- Décider, une fois plus de données CorBaMa disponibles, si le tagset fin (24
  catégories) doit remplacer notre tagset réduit à 11 classes en production.
- Comparer le BiLSTM à un modèle Transformer pré-entraîné (AfriBERTa,
  XLM-RoBERTa), une fois un jeu de test fiable établi.

## Auteur

Amadou H — stagiaire IA, RobotsMali.