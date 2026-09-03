
# 06 — Entraînement final, hyperparamètres et inférence

- [Boucle d&#39;entraînement/validation complète](final_training_and_inference.ipynb#full-training-loop)
- [Recherche d&#39;hyperparamètres](final_training_and_inference.ipynb#hyperparam-tuning)
- [Évaluation finale](final_training_and_inference.ipynb#final-evaluation)
- [Module d&#39;inférence](final_training_and_inference.ipynb#inference-module)

## Contenu

- `run_training()` : boucle avec class weights, early stopping et sauvegarde
  automatique du meilleur checkpoint (`models/bambara_pos_best.pth`).
- Recherche d'hyperparamètres sur `learning_rate`, `batch_size`, `hidden_dim`
  (12 configurations, entraînement court avec early stopping pour comparer).
- Évaluation finale sur le test set avec `classification_report`, matrice de
  confusion, et classement explicite des tags du plus difficile au plus facile
  (`per_tag_f1`).
- `inference.py` : module Python autonome (`BambaraPOSTagger`), utilisable en
  dehors de Jupyter, y compris en ligne de commande.

## Limites à garder en tête

Les scores de ce module héritent du diagnostic de data leakage du notebook 05 :
une partie des tags (`AUX`, `PRON`, `POSTP`, en partie `VERBE`) sont couverts
par le lexique ou la règle contextuelle du notebook 04, donc l'accuracy globale
reste optimiste. Le classement des tags les plus difficiles reste néanmoins
utile pour prioriser un futur travail d'annotation humaine.

## Utilisation en ligne de commande

```bash
cd 06_final_training_and_inference
python inference.py "Amadou bɛ kalan kɛ ."
```
