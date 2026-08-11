# Rapport Hebdomadaire : Semaines 7-8

**Projet :** Étiquetage Grammatical (POS Tagging) pour la langue Bambara
**Période :** Août 2026
**Étudiant :** Amadou H. TRAORE

---

## 1. Objectifs des Semaines 7-8

- Harmonisation du jeu de données selon la nomenclature formelle des étiquettes POS (`PRON`, `AUX`, `NOM`, `VERBE`, `POSTP`, `CONJ`, `ADV`, `PUNCT`).
- Implémentation du pipeline de prétraitement avec résolution automatique des contractions courantes (*k'a* en *ka a*).
- Ré-entraînement et validation du modèle **BiLSTM** bidirectionnel sous PyTorch.
- Fine-tuning d'un modèle Transformer pré-entraîné multilingue (**XLM-RoBERTa**).
- Analyse comparative des deux architectures sur le corpus annoté.

---

## 2. Tableau Comparatif des Architectures

| Critère d'Évaluation                      | Modèle BiLSTM (PyTorch)                      | Modèle XLM-RoBERTa (Transformer)               |
| :------------------------------------------ | :-------------------------------------------- | :---------------------------------------------- |
| **Perte finale (Loss)**               | **~0.02**                               | **~0.43**                                 |
| **Vitesse d'entraînement**           | Rapide ($\approx$ 2 secondes sur GPU)       | Modérée ($\approx$ 20-30 secondes)          |
| **Nombre de paramètres**             | Léger ($\approx$ 500k paramètres)         | Lourd ($\approx$ 270 millions de paramètres) |
| **Traitement du vocabulaire**         | Basé sur le dictionnaire de tokens explicite | Basé sur les subtokens (Subword Tokenization)  |
| **Généralisation (Mots rares/OOV)** | Limitée au vocabulaire vu                    | Forte (grâce au transfert d'apprentissage)     |
| **Précision en Inférence**          | 100% sur le jeu de test réduit               | 100% sur le jeu de test réduit                 |

---

## 3. Prédictions Comparées en Inférence

**Phrase de test :** `Amadou bɛ kalan kɛ so kɔnɔ.`

```text
TOKEN           | TAG BiLSTM      | TAG XLM-ROBERTA | ÉTAT ACCORD
---------------------------------------------------------------
Amadou          | NOM             | NOM             | ✅ Conforme
bɛ              | AUX             | AUX             | ✅ Conforme
kalan           | NOM             | NOM             | ✅ Conforme
kɛ              | VERBE           | VERBE           | ✅ Conforme
so              | NOM             | NOM             | ✅ Conforme
kɔnɔ            | POSTP           | POSTP           | ✅ Conforme
.               | PUNCT           | PUNCT           | ✅ Conforme
```

---

## 4. Conclusion & Prochaines Étapes

Les deux modèles atteignent une précision parfaite sur la structure grammaticale testée. Le BiLSTM offre une empreinte mémoire très faible et un temps d'exécution rapide, tandis que XLM-RoBERTa apporte une robustesse théorique supérieure face aux mots inconnus (*Out-Of-Vocabulary*).
