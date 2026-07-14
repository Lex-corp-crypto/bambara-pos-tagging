# Rapport Hebdomadaire - Semaine 1
**Période :** Juillet 2026  
**Étudiant :** Amadou H. TRAORE  

## 1. Tâches Réalisées Cette Semaine
* **Environnement & Outils :** Configuration de l'environnement virtuel Python, initialisation du dépôt Git local et liaison avec GitHub.
* **Concepts théoriques :** Assimilation des bases du POS tagging (Embeddings -> LSTM -> Couche Linéaire -> LogSoftmax).
* **Pratique & Code :** Écriture et entraînement réussi d'un premier "Toy Model" d'étiquetage morphosyntaxique avec PyTorch dans un notebook Jupyter. Le modèle a convergé avec une perte d'environ 0.2.

## 2. Difficultés Rencontrées & Solutions
* **Gestion des mots inconnus (OOV) :** Observation du comportement du token `<UNK>` face au mot non vu "ordinateur". Le modèle tend à lui attribuer l'étiquette par défaut en fonction de son apprentissage minimal, ce qui souligne l'importance d'un dictionnaire d'entraînement plus vaste ou de plongements de caractères.

## 3. Objectifs pour la Semaine Prochaine
* Passer à la phase d'évaluation sur un jeu de données de référence public (UD POS Tagging / Treebank) sur Kaggle.
* Mettre en place un `DataLoader` PyTorch robuste pour gérer le rembourrage (*padding*) et les dimensions des séquences de tailles variables.