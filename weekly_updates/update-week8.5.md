
# Rapport Hebdomadaire - Semaines 8.5

**Période :** Août 2026
**Projet :** Étiquetage Morphosyntaxique (POS Tagging) du Bambara via PyTorch (Lexique Étendu & Rééquilibrage)
**Étudiant :** Amadou H. TRAORE

---

## 1. Objectif de la semaine 8.5 : Extension à 12 Tags et Gestion du Déséquilibre de Classes

L'objectif principal de cette phase était de résoudre l'effondrement des classes minoritaires identifié lors des semaines précédentes. Nous avons réintroduit la grammaire étendue du Bambara (**12 tags**) et appliqué une stratégie de pondération dynamique de la perte (*Class Weights*) couplée à un ré-étiquetage lexical pour stabiliser la prédiction des mots fonctionnels.

---

## 2. Travaux Réalisés & Performance du Modèle

* **Enrichissement Lexical & Découpage du Corpus :** Ingestion du fichier `bambara_pos_prep.conll` (37 392 phrases) ré-étiqueté via un dictionnaire de règles grammaticales (`BAMBARA_EXTENDED_LEXICON`) couvrant 12 classes (`ADJ`, `ADV`, `AUX`, `CONJ`, `DET`, `NOM`, `PART`, `POSTP`, `PRON`, `PUNCT`, `VERBE`, `<PAD>`).
* **Optimisation de l'Architecture BiLSTM (`LSTMTaggerWithBatch`) :**
  * Modèle BiLSTM à 2 couches (`hidden_dim = 256`, `embedding_dim = 128`).
  * Ajout d'une couche de régularisation `Dropout(0.3)` et optimiseur `AdamW` (`lr = 0.001`, `weight_decay = 0.01`).
  * Intégration des poids de classe inversés (`compute_class_weight`) directement dans la fonction de perte `nn.NLLLoss(weight=class_weights)`.
* **Résultats d'Évaluation sur Corpus Enrichi (10 Époques) :**
  * **Meilleure Exactitude en Validation (Dev Acc) :** **99,78%** (Époque 10).
  * **Exactitude Finale sur le Jeu de Test (Test Acc) :** **99,85%** (Loss Test minimale).
  * **Couverture Parfaite des Classes Rares :** Les catégories à faible effectif (`ADV`, `NOM`, `PART`, `VERBE`) atteignent un F1-score de **1,00** sur le test set enrichi.

---

## 3. Difficultés Rencontrées & Diagnostic (Data Leakage & Corpus Brut)

* **Problème de Fuyotage de Données (*Data Leakage*) :**
  * Le score exceptionnel de **99,85%** est dû au fait que le lexique a ré-étiqueté de façon déterministe les ensembles Train, Dev et Test avant la séparation des données. Le modèle a principalement mémorisé la cartographie lexicale fixe.
* **Évaluation Réelle sur Corpus Brut (`bambara_pos_prep.conll`) :**
  * L'évaluation du modèle pondéré sur le fichier d'origine (sans surcouche de dictionnaire) donne une **Exactitude réelle de 57,01%**.
  * **Cause identifiée :** Le dataset CoNLL d'origine souffre d'un fort bruit d'annotation (absence totale de tags comme `AUX` ou `POSTP` dans le ground truth initial et sur-représentation de la classe `ADJ` avec 57 592 tokens).

---

## 4. Statut Actuel et Perspectives

Le pipeline d'entraînement rééquilibré est totalement opérationnel et l'exportation du modèle (`bambara_pos_weighted.pth`) et des règles (`bambara_tag_mappings_weighted.json`) est finalisée.

**Perspectives pour la suite :**

* **Pipeline d'Inférence Hybride :** Mettre en place une fonction de prédiction à 3 niveaux (Filtrage strict des ponctuations $\rightarrow$ Verification dans le lexique $\rightarrow$ Prédiction par le modèle BiLSTM).
* **Nettoyage du Dataset Ground Truth :** Formaliser la démarche de *Bootstrapping* lexical pour valider l'extension à 12 tags de façon scientifiquement rigoureuse.
