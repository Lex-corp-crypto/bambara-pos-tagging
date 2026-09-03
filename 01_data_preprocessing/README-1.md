# Module 01 : Prétraitement des Données & Tokenisation (Data Preprocessing)

Ce module gère le chargement, la tokenisation et la vectorisation des phrases pour la tâche d'étiquetage morphosyntaxique (POS Tagging) en Bambara.

**Étudiant :** Amadou H. TRAORE

---

## 💡 Fonctionnalités Principales

* **Indexation des Vocabulaires (`word_to_ix` & `tag_to_ix`) :** Conversion des chaînes de caractères en indices numériques acceptés par PyTorch.
* **Gestion des Jetons Spéciaux :**
  * `<PAD>` (Index 0) : Permet de normaliser la longueur des séquences dans les batchs.
  * `<UNK>` (Index 1) : Gère les mots hors-vocabulaire (OOV) lors de l'inférence.
* **Tokenisation Spécifique au Bambara :** Prise en compte des caractères de l'alphabet national du Mali (`ɛ`, `ɔ`, `ɲ`, `ŋ`).

---

## 🔗 Accès Direct au Code (Notebook Links)

Vous pouvez naviguer directement vers les étapes clés du notebook :

1. [Configuration de l&#39;Environnement](data_preprocessing.ipynb#env-setup)
2. [Création des Dictionnaires &amp; Vocabulaire](data_preprocessing.ipynb#vocab-mapping)
3. [Conversion en Tenseurs PyTorch (`prepare_sequence`)](data_preprocessing.ipynb#prepare-sequence)
4. [Tokenisation Adaptée au Bmbara](data_preprocessing.ipynb#tokenization)

## Exemple Rapide

```python
from data_preprocessing import bamanankan_tokenizer, prepare_sequence

text = "Amadou bɛ kalan kɛ."
tokens = bamanankan_tokenizer(text)
# Résultat : ['Amadou', 'bɛ', 'kalan', 'kɛ', '.']
```
