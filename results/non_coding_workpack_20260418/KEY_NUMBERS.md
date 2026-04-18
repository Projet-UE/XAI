# Key Numbers (Ready To Cite)

## autoPET FDG (main line)

Main tracked state: `post_best_dice_50epochs`

| Metric | Value |
|---|---:|
| Case count | 7 |
| Mean Dice | 0.4867 |
| Mean false negative volume (ml) | 41.2100 |
| Mean false positive volume (ml) | 6.2934 |

Tradeoff vs raw 50-epoch:

| Comparison | Dice delta | FN volume delta (ml) | FP volume delta (ml) |
|---|---:|---:|---:|
| best_dice_vs_raw | +0.1815 | +5.5416 | -24.1622 |
| low_fp_vs_raw | +0.0691 | +4.1180 | -29.1849 |

XAI protocol (tracked state):

| Method | Rank | Protocol score |
|---|---:|---:|
| integrated_gradients | 1 | 0.0239 |

## Brain MRI (backup line)

| Metric | Value |
|---|---:|
| Accuracy | 0.8684 |
| Precision | 0.9091 |
| Recall | 0.8696 |
| F1 | 0.8889 |
| ROC-AUC | 0.9391 |
| Threshold | 0.50 |

Confusion matrix:

| Actual \\ Predicted | No | Yes |
|---|---:|---:|
| No | 13 | 2 |
| Yes | 3 | 20 |

Brain MRI XAI benchmark ranking (confidence-drop protocol):

| Rank | Method | Mean confidence drop | 95% CI (low, high) |
|---:|---|---:|---|
| 1 | occlusion | 0.0465 | [0.0106, 0.0837] |
| 2 | gradcam | 0.0253 | [0.0084, 0.0455] |
| 3 | integrated_gradients | 0.0181 | [-0.0384, 0.0762] |
