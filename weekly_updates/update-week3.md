# Rapport Hebdomadaire - Semaine 3
**Période :** Juillet 2026  
**Étudiant :** Amadou H. TRAORE  

## 1. Tâches Réalisées : Industrialisation du Pipeline de Données
Après avoir validé les stratégies de segmentation en Semaine 2.5, l'objectif de cette semaine était de connecter notre tokeniseur aux structures de données natives de PyTorch pour automatiser l'entraînement par lots.

* **Création du `BambaraPOSDataset` (Version Française) :** Implémentation d'une classe personnalisée héritant de `torch.utils.data.Dataset`. Elle intègre désormais le processus de tokenisation propre en amont pour transformer les phrases de texte brut en listes de tokens, puis en tenseurs d'index.
* **Configuration du `DataLoader` :** Mise en place du chargement automatique avec mélange des données (`shuffle=True`) et définition d'une taille de lot (`batch_size=2`).
* **Fonction de Collation Dynamique (`collate_fn_padd`) :** Code d'une fonction sur-mesure qui intercepte les lots du DataLoader pour appliquer un *padding* (rembourrage de zéros) dynamique. La taille de la matrice s'ajuste automatiquement à la phrase la plus longue de *chaque lot* (ex: un lot à `[2, 5]` et un autre à `[1, 2]`), optimisant drastiquement l'usage de la mémoire.

## 2. Validation et Entraînement du Modèle Final
* **Gestion du Masquage :** Entraînement de notre modèle `LSTMTaggerWithBatch` en configurant la fonction de perte avec `ignore_index=0` afin que le modèle n'apprenne pas sur les jetons de rembourrage (`<PAD>`).
* **Convergence :** Validation de la boucle de rétropropagation sur 150 époques avec une décroissance fluide de la perte moyenne.
* **Tests Unitaires :** Déploiement d'une cellule de test interactive permettant d'évaluer le modèle sur de nouvelles phrases tokenisées, mettant en lumière le comportement du modèle face aux mots inconnus (`<UNK>`).

## 3. Objectifs pour la Semaine Prochaine
* Transition vers un jeu de données de taille réelle (Corpus annoté).
* Adaptation des expressions régulières ou des règles du tokeniseur pour basculer officiellement sur la morphologie de la langue Bambara.