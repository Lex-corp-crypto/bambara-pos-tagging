# Module 02 : Architecture des Modèles (Model Architecture)

Ce module regroupe les définitions des réseaux de neurones récurrents (PyTorch `nn.Module`) utilisés pour prédire les étiquettes morphosyntaxiques (POS tags).

**Étudiant :** Amadou H. TRAORE

## Modèles Disponibles

1. **`LSTMTagger` (Unidirectionnel) :**

   * Architecture de base pour traiter les phrases séquence par séquence.
   * Adaptée à des entrées simples sans gestion complexe du batching.
2. **`AdvancedBiLSTMTagger` (Bidirectionnel & Batching) :**

   * **Bidirectionnel (BiLSTM) :** Capture simultanément le contexte gauche et droit de chaque mot.
   * **Couche de Dropout :** Évite le surapprentissage (overfitting) durant l'entraînement.
   * **`batch_first=True` :** Optimisé pour le traitement par lots et le remplissage (padding).

---

## 🔗 Accès Direct au Code (Notebook Links)

1. [Définition de `LSTMTagger` de base](model_architecture.ipynb#intro-architecture)
2. [Définition du `AdvancedBiLSTMTagger`](model_architecture.ipynb#bilstm-architecture)
3. [Validation des dimensions et formes des tenseurs](model_architecture.ipynb#test-instantiation)

---

## Dimensions des Tenseurs (Tensor Shapes)

| Couche                   | Forme Entrée (Input Shape)               | Forme Sortie (Output Shape)               |
| :----------------------- | :---------------------------------------- | :---------------------------------------- |
| **Embedding**      | `(Batch Size, Seq Len)`                 | `(Batch Size, Seq Len, Embed Dim)`      |
| **BiLSTM**         | `(Batch Size, Seq Len, Embed Dim)`      | `(Batch Size, Seq Len, Hidden Dim * 2)` |
| **FC (Linéaire)** | `(Batch Size, Seq Len, Hidden Dim * 2)` | `(Batch Size, Seq Len, Tagset Size)`    |
