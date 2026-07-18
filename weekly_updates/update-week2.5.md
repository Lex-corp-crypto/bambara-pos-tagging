# Rapport Hebdomadaire - Semaine 2.5
**Période :** Juillet 2026  
**Étudiant :** Amadou H. TRAORE  

## 1. Directive du Superviseur sur la Tokenisation
Lors de mon point d'étape au laboratoire RobotsMali, Mr YACOUBA DIARRA a souligné qu'un simple découpage par `.split()` sur les espaces était insuffisant pour un pipeline de traitement automatique du langage (NLP) rigoureux. En français, les apostrophes (ex: *L'étudiant*), les traits d'union (ex: *travaille-t-il*) et la ponctuation collée aux mots créent des tokens erronés qui polluent le dictionnaire et génèrent un taux anormalement élevé de mots inconnus (`<UNK>`).

## 2. Travaux Réalisés : Étude et Comparaison de Tokeniseurs
Afin de choisir la meilleure stratégie de segmentation, j'ai implémenté et comparé trois approches distinctes de tokenisation appliquées à la langue française :

1. **Tokenisation par Expressions Régulières (Regex) :** Création d'un motif personnalisé pour isoler la ponctuation et capturer les élisions de base.
2. **Tokenisation via NLTK (`WordPunctTokenizer`) :** Utilisation d'une approche statistique classique qui sépare proprement tous les caractères alphabétiques des caractères non-alphabétiques (points, virgules, apostrophes).
3. **Tokenisation via SpaCy (`fr_core_news_sm`) :** Utilisation d'un modèle de Deep Learning industriel basé sur des règles linguistiques complexes permettant de traiter les structures idiomatiques françaises de manière optimale.

## 3. Analyse du Test Comparatif
Sur la phrase de test : *"L'étudiant travaille-t-il au laboratoire, aujourd'hui ?"*
* **NLTK** et **SpaCy** ont réussi à détacher l'apostrophe (`"L'"`, `"étudiant"`) et à isoler les signes de ponctuation (`","`, `"?"`).
* **SpaCy** s'est distingué par sa gestion avancée des traits d'union et des structures grammaticales complexes, ce qui en fait le candidat idéal pour nettoyer nos phrases avant de les transformer en tenseurs PyTorch.

## 4. Prochaine Étape (Semaine 3)
Intégration du tokeniseur sélectionné au sein de notre classe `BambaraPOSDataset` et automatisation complète via le `DataLoader` PyTorch.