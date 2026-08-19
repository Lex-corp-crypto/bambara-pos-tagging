# Rapport Hebdomadaire - Semaines 9 & 10
**Période :** Août 2026  
**Projet :** Étiquetage Morphosyntaxique (POS Tagging) du Bambara via PyTorch  
**Étudiant :** Amadou H. TRAORE  

## 1. Objectif des Semaines 9 & 10 : Passation à l'Échelle et Bidirectionnalité
L'objectif central de cette phase était de faire passer notre modèle d'étiquetage morphosyntaxique (POS Tagging) à l'échelle industrielle en exploitant l'intégralité du corpus annoté *Bayelemabaga* (**37 392 phrases**), tout en basculant l'architecture vers un LSTM bidirectionnel (**BiLSTM**) afin d'optimiser la capture du contexte linguistique en Bambara.

## 2. Travaux Réalisés & Architecture MLOps
* **Ingestion et Structuration à Grande Échelle :** Parsing du corpus complet `bambara_pos_prep.conll` et répartition stricte 80/10/10 entre les ensembles d'entraînement (train), de validation (dev) et de test (test).
* **Architecture BiLSTM (`LSTMTaggerWithBatch`) :** Passage à une couche LSTM bidirectionnelle (`bidirectional=True`) couplée à un embedding de dimension 128 et un état caché de dimension 256. Cela permet au modèle d'analyser simultanément le contexte gauche et droit de chaque token[cite: 4].
* **Pipeline d'Entraînement & Sécurisation MLOps :**
  * Alignement et rembourrage dynamique via `collate_fn_padd` et `pad_sequence`.
  * Masquage propre des jetons de rembourrage dans la perte (`nn.NLLLoss(ignore_index=0)`).
  * Implémentation d'un régulateur de gradient (`clip_grad_norm_` à 1.0) et d'un ajustement dynamique du taux d'apprentissage via `ReduceLROnPlateau`.
  * Sauvegarde automatique du meilleur modèle (`bambara_pos_best.pth`) et exportation des mappings au format JSON (`bambara_tag_mappings_final.json`).

## 3. Gestion des Difficultés & Résolution des Bugs
* **Linguistique - Contextualisme du Bambara :** Un LSTM unidirectionnel limitait la capture des postpositions ou marqueurs prédicatifs postérieurs au mot. L'activation de la bidirectionnalité a permis de doubler la capacité d'analyse contextuelle[cite: 4].
* **Dépaquetage des Batches (`ValueError: too many values to unpack`) :** La fonction `collate_fn_padd` renvoyant des tuples de 3 éléments `(inputs, targets, lengths)`, les boucles d'évaluation standards plantaient. Toutes les boucles d'entraînement et d'évaluation ont été réajustées pour capturer explicitement les 3 variables.
* **Format des Entrées Embedding (`TypeError: embedding()... must be Tensor, not str`) :** Des chaînes de caractères brutes passaient dans la couche d'embedding suite à une réinitialisation de session. La classe `BambaraPOSDataset` a été corrigée pour forcer la conversion des tokens texte en entiers (`torch.long`) via les dictionnaires `word_to_ix` et `label2id`.
* **Variables Extérieures Non Définies (`NameError`) :** La réinitialisation du kernel effaçait le dictionnaire `tag_to_ix_bambara` ou la classe du modèle. Le code a été restructuré dans un pipeline unique et autonome.

## 4. Statut Actuel et Perspectives
Le pipeline d'entraînement est désormais 100% stable, entraîné sur un corpus massif de 37k phrases et capable d'évaluer ses performances sur le jeu de test avec génération d'un rapport de classification complet (`classification_report`). La prochaine étape (Semaine 11) sera consacrée à l'implémentation de la fonction d'inférence directe pour la prédiction sur du texte brut Bambara en environnement de production.