# Rapport Hebdomadaire - Semaine 4
**Période :** Juillet 2026  
**Étudiant :** Amadou H. TRAORE  

## 1. Objectif de la Semaine : Transition vers la Langue Bambara
L'objectif central de cette semaine était de faire basculer notre pipeline de traitement automatique du langage (NLP) du français vers le Bambara (*Bamanankan*), conformément aux objectifs finaux de notre projet d'étude.

## 2. Travaux Réalisés & Adaptations Linguistiques
* **Configuration Unicode du Tokeniseur :** L'alphabet national du Mali (Décret n°159/PG-RM) utilisant des caractères spécifiques, le tokeniseur standard coupait les mots de manière erronée. J'ai mis à jour les expressions régulières en étendant la plage de capture aux glyphes Unicode du bloc Latin Étendu, intégrant explicitement les voyelles ouvertes (`ɛ`, `ɔ`) et les consonnes nasales/palatales (`ɲ`, `ŋ`), en minuscules comme en majuscules.
* **Création du premier Jeu de Données Bambara :** Implémentation d'un corpus d'apprentissage initial tokenisé automatiquement avec succès (ex: *"Amadou bɛ kalan kɛ ."*).

## 3. Gestion et Résolution d'une Crise d'Architecture (Bug PyTorch)
Lors du passage des nouvelles données dans la boucle d'entraînement, une erreur critique de dimensionnement est survenue (`ValueError: Expected input batch_size to match target batch_size`).
* **Diagnostic :** Un effet mémoire dans l'environnement Jupyter conservait les dimensions de l'ancien DataLoader (français), créant un décalage asymétrique lors du calcul de la perte (`NLLLoss`) avec les nouveaux dictionnaires d'index.
* **Résolution :** Industrialisation du code dans un bloc d'exécution unique assurant la reconstruction synchrone des dictionnaires (`word_to_ix` et `tag_to_ix`), l'instanciation du `BambaraPOSDataset` et la configuration dynamique de la perte via `.view(-1, NB_TAGS_CLASSES)`. Un test de sécurité (`assert`) a également été intégré pour valider l'alignement strict entre la longueur des phrases et celle des étiquettes grammaticales.

## 4. Statut Actuel et Perspectives
Le modèle `LSTMTaggerWithBatch` converge de manière optimale sur le jeu de données en Bambara et est capable de prédire correctement les étiquettes de test. La prochaine étape (Semaine 5) sera dédiée à l'industrialisation par le chargement d'un véritable corpus de grande taille via un fichier externe (`.txt` / `.tsv`).