# Rapport Hebdomadaire - Semaine 2
**Période :** Juillet 2026  
**Étudiant :** Amadou H. TRAORE  

## 1. Tâches Réalisées Cette Semaine
* **Gestion du Padding :** Implémentation de la fonction `pad_sequence` de PyTorch pour uniformiser la taille de phrases de longueurs différentes dans un même tenseur.
* **Architecture par Lots (Batches) :** Adaptation de notre modèle avec la classe `LSTMTaggerWithBatch` pour traiter des lots de données en parallèle (`batch_first=True`).
* **Optimisation de l'Embedding :** Utilisation du paramètre `padding_idx=0` pour s'assurer que les jetons de rembourrage ne perturbent pas l'apprentissage des représentations de mots.

## 2. Difficultés Réalisées & Solutions
* **Changement de dimensions (Shapes) :** Le passage d'un traitement phrase par phrase (dimension `[longueur_phrase, 1, embedding_dim]`) à un traitement par batch (dimension `[taille_batch, longueur_phrase, embedding_dim]`) a demandé une réorganisation de la fonction `forward` du modèle, résolue en utilisant correctement les paramètres de lot intégrés à PyTorch.

## 3. Objectifs pour la Semaine Prochaine
* Commencer l'exploration de l'arène Kaggle avec un jeu de données de référence public (comme UD POS Tagging).
* Écrire un `DataLoader` complet et robuste pour automatiser ce processus de batching et de padding sur de grands volumes de données.