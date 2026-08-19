# Rapport Hebdomadaire - Semaines 9 & 10

**Période :** Août 2026
**Projet :** Étiquetage Morphosyntaxique (POS Tagging) du Bambara via PyTorch
**Étudiant :** Amadou H. TRAORE

## 1. Objectif des Semaines 9 & 10 : Passage à l'Échelle et Modélisation BiLSTM

L'objectif principal de cette phase était d'entraîner notre modèle d'étiquetage morphosyntaxique sur l'intégralité du corpus annoté *Bayelemabaga* (**37 392 phrases**), en migrant vers une architecture LSTM bidirectionnelle (**BiLSTM**) optimisée pour capturer les dépendances contextuelles bidirectionnelles du Bambara.

## 2. Travaux Réalisés & Performance du Modèle

* **Ingestion et Découpage du Corpus :** Ingestion complète du fichier `bambara_pos_prep.conll` (37 392 phrases) avec répartition 80/10/10 (Train: 29 913 | Dev: 3 739 | Test: 3 740 phrases).
* **Architecture BiLSTM (`LSTMTaggerWithBatch`) :** Embedding de dimension 128, couche LSTM bidirectionnelle de dimension cachée 256, et optimiseur Adam (lr = 0.001) couplé à un scheduler `ReduceLROnPlateau`.
* **Résultats d'Évaluation :**
  * **Meilleure Exactitude en Validation (Dev Acc) :** **92,84%** (Époque 3)
  * **Exactitude Finale sur le Jeu de Test (Test Acc) :** **95,09%** (Loss Test = 0.1893)
  * **Couverture des Noms et Ponctuations :** F1-score de **0,97** sur la classe majoritaire `NOM` et **0,93** sur `PUNCT`.

## 3. Difficultés Rencontrées & Diagnostic

* **Erreurs de Code Résolues :**
  * Correction du dépaquetage de `DataLoader` (`for inputs, targets, lengths in test_loader`) lié à `collate_fn_padd`.
  * Résolution des types de données dans l'Embedding via le réalignement de `BambaraPOSDataset`.
* **Problème de Déséquilibre des Classes (*Class Imbalance*) :**
  * Le dictionnaire retenu sur ce run comporte 6 classes (`ADJ`, `NOM`, `PRON`, `PUNCT`, `VERBE`, `<PAD>`).
  * Les classes minoritaires comme `ADJ` (145 tokens) et `PRON` (940 tokens) n'ont pas été prédites par le modèle (F1-score = 0.00), masquées par le poids numérique massif des `NOM` (57 592 tokens).

## 4. Statut Actuel et Perspectives

Le pipeline global est fonctionnel, rapide (exécuté sur CPU) et atteint une exactitude globale remarquable de **95,09%**.

**Perspectives pour la Semaine 11 :**

* **Rééquilibrage des Poids de Perte (*Class Weights*) :** Intégrer `weight` dans `nn.NLLLoss` pour forcer le modèle à pénaliser les erreurs sur les classes rares (`ADJ`, `PRON`).
* **Réintégration des Tags Étendus :** Corriger le filtre pour réintroduire les marqueurs auxilaires (`AUX`), postpositions (`POSTP`) et déterminants (`DET`).
