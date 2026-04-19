# Interpretation synthétique

## autoPET FDG (ligne principale)

- Le snapshot principal confirme une segmentation mesurée par Dice `0.4867` avec FP moyen `6.2934` mL et FN moyen `41.2100` mL.
- Méthode XAI retenue en priorité sur autoPET: `integrated_gradients`.
- Le benchmark XAI autoPET multi-méthodes repose sur un run reconstruit traqué séparément ; ses métriques propres sont copiées dans `autopet/xai_segmentation_metrics.json`.
- Variante best-Dice vs raw: Dice +0.1815, FN +5.5416 mL, FP -24.1622 mL.
- Variante low-FP vs raw: Dice +0.0691, FN +4.1180 mL, FP -29.1849 mL.

## Brain MRI (ligne secondaire)

- La ligne secondaire Brain MRI reste cohérente pour la soutenance: accuracy `0.8684`, F1 `0.8889`, ROC-AUC `0.9391`.
- Méthode XAI retenue en priorité sur Brain MRI: `occlusion`.

## Lecture d'ensemble

- Positionnement du projet: autoPET FDG est la contribution scientifique principale; Brain MRI confirme la robustesse de la démarche XAI sur un second cadre.
- Le dossier met en évidence le compromis Dice/FN/FP des variantes post-traitées plutôt qu'un unique score isolé.
- Les méthodes XAI sont interprétées comme explications de décision du modèle, pas comme preuve clinique directe.
- Une galerie XAI élargie (16 cas équilibrés) est disponible dans `results/brain_mri_refresh_xai_20260418/`.
- Le benchmark Brain MRI utilisé dans ce pack est `results/brain_mri_xai_benchmark_20260418_clean_full/`.
