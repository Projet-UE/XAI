# Interpretation synthétique

## autoPET FDG (ligne principale)

- Le snapshot principal confirme une segmentation mesurée par Dice `0.4867` avec FP moyen `6.2934` mL et FN moyen `41.2100` mL.
- Méthode XAI prioritaire sur autoPET: `integrated_gradients`.
- Variante best-Dice vs raw: Dice +0.1815, FN +5.5416 mL, FP -24.1622 mL.
- Variante low-FP vs raw: Dice +0.0691, FN +4.1180 mL, FP -29.1849 mL.

## Brain MRI (ligne backup)

- Le backup Brain MRI reste cohérent pour la soutenance: accuracy `0.8684`, F1 `0.8889`, ROC-AUC `0.9391`.
- Méthode XAI prioritaire sur Brain MRI: `integrated_gradients`.

## Message projet recommandé

- Storyline: autoPET FDG est la contribution scientifique principale; Brain MRI confirme la robustesse de la démarche XAI sur un second cadre.
- Le dossier met en évidence le compromis Dice/FN/FP des variantes post-traitées plutôt qu'un unique score isolé.
- Les méthodes XAI sont interprétées comme explications de décision du modèle, pas comme preuve clinique directe.
- Une galerie XAI élargie (16 cas équilibrés) est disponible dans `results/brain_mri_refresh_xai_20260418/`.
- Un benchmark rapide de contrôle est disponible dans `results/brain_mri_xai_benchmark_20260418_clean_fast/`.
