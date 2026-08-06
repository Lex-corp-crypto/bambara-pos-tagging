# Rapport Hebdomadaire - Semaine 6
**Période :** Août 2026  
**Étudiant :** Amadou H. TRAORE  

## 1. Objectif de la Semaine : Pseudo-Labelling & Entraînement Massif du Modèle POS
L'objectif central de cette semaine était de surmonter la rareté des données annotées en étiquetage morphosyntaxique (POS Tagging) Bambara en déployant une stratégie d'annotation automatique (*bootstrapping / pseudo-labelling*) sur le corpus de traduction `bayelemabaga`, puis d'entraîner notre modèle LSTM final sur un grand volume de données.

## 2. Travaux Réalisés & Pseudo-Labelling
* **Correction des Dictionnaires & Modèle Semence (*Seed Model*) :** Entraînement d'un premier modèle LSTM Tagger sur le corpus initial annoté à la main (`corpus_bambara.tsv`) pour constituer un ensemble minimal d'étiquettes grammaticales réelles (`NOM`, `VERBE`, `AUX`, `PUNCT`, etc.).
* **Pipeline d'Annotation Massive :** Déploiement du script d'inférence automatique `annotate_large_corpus` sur **37 392 phrases** du corpus brut `bayelemabaga`.
* **Exportation au Format Standard :** Génération et validation d'un nouveau fichier CoNLL annoté (`bambara_pos_prep.conll`) contenant plus de 37k phrases étiquetées au format `Mot \t Tag`.

## 3. Entraînement et Performances du Modèle Final
* **Répartition du Grand Corpus :** Découpage strict du dataset global au format 80% Entraînement (29 913 phrases), 10% Validation/Dev (3 739 phrases) et 10% Test (3 740 phrases).
* **Architecture du Modèle Final (`LSTMTaggerWithBatch`) :**
  * Dimension d'Embedding : 128
  * Dimension Cachée (LSTM) : 256
  * Optimiseur : Adam (lr = 0.001, Loss = `NLLLoss` avec `ignore_index=0`)
  * Taille de Lot (*Batch Size*) : 32
* **Résultats & Convergence :** Décroissance fluide de la perte d'entraînement de `0.0671` à `0.0002` sur 10 époques.
  * **Précision Validation (Val Accuracy) :** **99,97%**
  * **Précision Test (Test Accuracy) :** **99,97%** (Perte de Test = `0.0014`)

## 4. Analyse Technique & Sauvegarde
* **Inférence Interactive :** Validation des prédictions sur des phrases de test en Bamanankan (ex: *"Amadou bɛ kalan kɛ so kɔnɔ ."*), confirmant l'apprentissage réussi de la syntaxe et la bonne gestion de la casse.
* **Sauvegarde des Artefacts :** Exportation des poids du modèle final (`bambara_pos_lstm_final.pth`) et de la sérialisation des vocabulaires/index (`vocab_and_tags_final.pkl`).

## 5. Perspectives pour la Semaine Prochaine
* **Raffinement Linguistique du Dictionnaire de Tags :** Augmenter la granularité des étiquettes attribuées par le modèle semence pour intégrer les postpositions (`POSTP`), adjectifs (`ADJ`) et pronoms spécifiques.
* **Intégration d'Embeddings de Sous-mots (Subwords / Char-LSTM) :** Améliorer la généralisation sur les mots rares ou inconnus (*Out-Of-Vocabulary*) en capturant la morphologie agglutinante du Bambara.