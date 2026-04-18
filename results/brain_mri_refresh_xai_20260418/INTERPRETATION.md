# Interprétation concise du lot Brain MRI XAI (16 cas)

## Périmètre

- Lot qualitatif: `16` images test (`8 yes`, `8 no`).
- Méthodes visualisées dans ce dossier: `gradcam`, `integrated_gradients`.
- Modèle utilisé: checkpoint refresh Brain MRI (`ResNet18` binaire).
- Objectif: fournir un matériau lisible pour le rapport (pas une validation clinique).

## Comment lire les cartes XAI

- Les zones les plus intenses (rouge/jaune) indiquent les régions qui ont le plus influencé la décision du modèle.
- Une zone rouge ne signifie pas automatiquement “tumeur”; cela signifie “zone importante pour la prédiction”.
- La lecture correcte est: *importance pour la décision du réseau*, pas *preuve médicale directe*.

## Ce que montre ce lot

- La couverture est plus large que la galerie initiale (4 cas), ce qui réduit le risque de conclure sur des exemples trop favorables.
- Les probabilités du lot sont globalement cohérentes avec les classes:
  - moyenne `yes`: `0.5390`
  - moyenne `no`: `0.3901`
- Cas limites (proba proche de 0.5) observés dans ce lot:
  - `no/4 no.jpg` (`0.5052`)
  - `no/10 no.jpg` (`0.5207`)
  - `yes/Y71.JPG` (`0.4998`)
  - `yes/Y148.JPG` (`0.5153`)
  - `yes/Y27.jpg` (`0.5385`)

## Lecture méthode par méthode (niveau projet)

- `integrated_gradients` produit des cartes plus concentrées et généralement plus informatives visuellement sur les cas positifs.
- `gradcam` reste utile pour une vue macro rapide, mais les cartes sont plus diffuses sur les cas ambigus.
- Dans le benchmark quantitatif du projet (confidence-drop), `integrated_gradients` est aussi la méthode la mieux classée, ce qui est cohérent avec la lecture visuelle.
- Un sanity-check rapide relancé sur ce lot propre (2 méthodes, 500 bootstrap) est aussi disponible:
  - `results/brain_mri_xai_benchmark_20260418_clean_fast/`
  - Sur ce run rapide, `gradcam` passe légèrement devant `integrated_gradients`, avec des intervalles de confiance qui se recouvrent.
  - Interprétation: ce run sert surtout de contrôle de cohérence pipeline; la décision finale méthode doit rester basée sur le benchmark complet.

## Message prêt à réutiliser dans le rapport

- “Nous avons élargi la galerie Brain MRI XAI à 16 cas équilibrés pour éviter une conclusion basée sur un petit échantillon.  
  Les cartes XAI confirment que les régions mises en avant sont des régions de décision du modèle, et non une annotation clinique.  
  Integrated Gradients reste la méthode la plus convaincante dans notre protocole, tandis que les cas proches de 0.5 montrent les limites attendues du classifieur.”

## Limites explicites

- Ce lot reste qualitatif; il sert à expliquer les décisions du modèle, pas à valider une causalité clinique.
- Le modèle refresh (`3` epochs CPU) a des métriques inférieures au snapshot principal Grenoble; ce lot sert surtout d’illustration XAI étendue.
