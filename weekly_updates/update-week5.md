# Rapport Hebdomadaire - Semaine 5 & 6
Période : Juillet 2026
Étudiant : Amadou H. TRAORE

## 1. Objectif de la Semaine : Industrialisation du Corpus & Normalisation Texte
Dans la continuité des travaux de la Semaine 4, l'objectif principal de cette phase était de concrétiser l'étape d'industrialisation en ingérant, nettoyant et structurant un véritable corpus de grande taille en Bambara (Bamanankan) issu du jeu de données `bayelemabaga` de RobotsMali.

## 2. Travaux Réalisés & Traitement du Corpus
- Aggregation du Corpus Brut : Extraction et fusion des données bilingues et monolingues (`train.bam`, `dev.bam`, `test.bam`, `bambara.clean.txt`) en un flux unique.
- Pipeline de Normalisation & Segmentations : Implémentation de fonctions de nettoyage sur mesure (`normalize_bambara_text`) assurant la suppression des caractères parasites[cite: 1], la gestion des espaces ainsi que l'isolement strict des frontières de ponctuation (`.,!?;:'"`) pour éviter les collages de tokens lors du découpage.
- Pre-annotation Lexicale / Tagset de Base : Intégration d'un dictionnaire grammatical fonctionnel (pronoms *a, i, an*, marques verbales *bɛ, tɛ, ye, ka*, postpositions, etc.)[cite: 1] permettant de pré-annoter automatiquement le corpus au format standard `.conll` / `.tsv`.

## 3. Structuration pour PyTorch & Encodage des Sous-mots (Subwords)
Pour préparer le passage aux architectures profondes et aux modèles de langue pré-entraînés (ex: XLM-RoBERTa) :
- Alignement Tokens <-> Étiquettes : Implémentation de la classe custom `BambaraPOSDataset` gérant l'alignement entre les découpages en subwords du tokeniseur et le découpage grammatical mot-à-mot.
- Masquage de la Perte (-100) : Attribution de l'index d'ignorance `-100` sur les jetons de rembourrage (padding) et les fragments de sous-mots secondaires afin d'exclure ces positions du calcul de la `CrossEntropyLoss`.
- Validation des Tenseurs : Vérification synchrone des dimensions de lots (*batch dimensions*) obtenues via le `DataLoader` (formes strictes `[16, 64]` pour `input_ids`, `attention_mask` et `labels`).

## 4. Statut Actuel et Perspectives
Le pipeline d'ingestion et de préparation de données est 100% fonctionnel et prêt pour l'entraînement[cite: 1]. La prochaine étape (Semaines 7 & 8) consistera à affiner les étiquettes avec la validation d'experts linguistiques (DNFLN / Nouhoum) et à lancer la première boucle d'entraînement du modèle séquentiel complet.