# Module 04 : Pipeline POS Tagging pour le Bambara & Traitement du Corpus Bayelemabaga

Ce module est le cœur d'application du projet. Il gère l'intégration du jeu de données d'apprentissage **Bayelemabaga** (corpus annoté en Bambara/Bamanankan comprenant **37 392 phrases**) et fournit le pipeline complet d'inférence pour réaliser l'étiquetage morphosyntaxique automatisé sur du texte brut.

---

## 🎯 Objectifs du Module

1. **Extraction & Formatage du Corpus Bayelemabaga :**
   * Prise en charge des fichiers structurés au format TSV (`mot \t étiquette`).
   * Reconstruction dynamique des séquences de phrases et alignement des tokens avec leurs catégories grammaticales.
2. **Construction Automatique du Vocabulaire Réel :**
   * Génération de `word_to_ix` intégrant la gestion des tokens de remplissage (`<PAD>`) et des mots inconnus (`<UNK>`).
   * Indexation complète des catégories morphosyntaxiques (`tag_to_ix` et `ix_to_tag`).
3. **Pipeline d'Inférence End-to-End :**
   * Tokenisation adaptée au Bambara (intégration des graphes spécifiques : `ɛ`, `ɔ`, `ɲ`, `ŋ`).
   * Passage en modèle Deep Learning (BiLSTM) et prédiction des tags POS avec décodage `ArgMax`.

---

## 📋 Spécificités du Corpus Bayelemabaga

Le corpus **Bayelemabaga** est l'un des jeux de données de référence pour le Traitement Automatique du Langage (TAL / NLP) sur le Bambara.

### Jeu d'Étiquettes POS Courantes (Tagset) :

| Tag      | Signification                             | Exemple                       |
| :------- | :---------------------------------------- | :---------------------------- |
| `n`    | Nom (Noun)                                | *kalan*, *dugu*, *gafe* |
| `v`    | Verbe (Verb)                              | *kɛ*, *fɔ*, *taː*    |
| `pm`   | Marqueur Prédicatif (Predicative Marker) | *bɛ*, *tɛ*, *ye*      |
| `pers` | Pronom Personnel (Personal Pronoun)       | *A*, *I*, *An*          |
| `pp`   | Postposition                              | *kɔnɔ*, *la*, *fe*    |
| `adj`  | Adjectif (Adjective)                      | *ɲuman*, *belebele*      |
| `punc` | Ponctuation                               | `.`, `!`, `?`           |

---


**Section 1 : Chargement et Parsing du Corpus**

Le code extrait les données structurées du fichier `bambara_pos_prep_retagged.conll` pour alimenter le pipeline[cite: 14].

* **Format d'entrée** : Structure standard CoNLL / TSV (`mot \t tag`) où chaque ligne vide délimite la fin d'une phrase[cite: 14].
* **Algorithme de lecture** : Un itérateur de lignes charge le corpus en mémoire sous forme de liste de structures `(mots, tags)` tout en appliquant une assertion de vérification de longueur pour éviter tout désalignement entre les jetons et leurs étiquettes grammaticales respectives[cite: 14].

---

**Section 2 : Métriques et Données Statistiques**

Le traitement analytique produit un aperçu volumétrique et synthétique du corpus[cite: 14] :

| Métrique                      | Valeur extraite |
| :----------------------------- | :-------------- |
| **Phrases totales**      | 37 392          |
| **Tokens totaux**        | 639 614         |
| **Tags distincts (PoS)** | 11              |

**Distribution des Catégories Grammaticales (PoS)**

* **NOM** : 184 133
* **PUNCT** : 104 611
* **AUX** : 97 075
* **PRON** : 91 797
* **VERBE** : 63 984
* **POSTP** : 33 411
* **CONJ** : 27 729
* **DET** : 24 362
* **PART** : 7 486
* **ADV** : 2 599
* **ADJ** : 2 427

---

**Section 3 : Inspecteur de Données (Exemple)**

Structure du premier échantillon du corpus (Index 0):

```python
Mots : ['Mieru', 'Baa', 'ka', 'maana', '.', 'Ayiwa', '!']
Tags : ['NOM',   'NOM', 'AUX', 'VERBE', 'PUNCT', 'NOM', 'PUNCT']
```
