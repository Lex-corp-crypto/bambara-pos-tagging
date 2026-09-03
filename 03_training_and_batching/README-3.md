
# Module 03 : Entraînement et Batching (Training & Batching)

Ce module gère le découpage des données en lots (batches), l'alignement de la longueur des séquences (padding) et l'exécution de la boucle d'apprentissage automatique avec PyTorch.

---

## ⚙️ Composants Majeurs

1. **`POSDataset` & `pad_collate_fn` :**
   * Encapsule le jeu de données pour PyTorch.
   * Utilise `pad_sequence` pour égaliser dynamiquement la longueur des phrases dans chaque lot avec la valeur `0` (`<PAD>`).
2. **Gestion de la Perte (`nn.CrossEntropyLoss`) :**
   * Utilisation de l'argument `ignore_index=0` pour que la perte ne soit pas calculée sur les jetons de remplissage (`<PAD>`).

---

## 🔗 Accès Direct au Code (Notebook Links)

1. 📦 [Création du Dataset et du Collate DataLoader](training_batching.ipynb#dataset-dataloader)
2. 🔄 [Définition de la Boucle d&#39;Entraînement](training_batching.ipynb#training-loop)
3. 🧪 [Test complet sur des données d&#39;exemple](training_batching.ipynb#pipeline-test)

---

## 📊 Flux de Données (Batch Processing Pipeline)

```text
Phrases brutes -> Indexation -> Padding (pad_collate_fn) -> BiLSTM -> Loss (ignore <PAD>) -> Backpropagation
```
