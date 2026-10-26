# Model results summary

These metrics are from the recorded local run, using a stratified 80/20 train/test split, a class-weighted Random Forest, and a 0.50 prediction threshold. Small differences can occur with library versions and reruns.

## Five-fold training cross-validation

- Mean ROC-AUC: 0.776
- Mean average precision: 0.359
- Mean precision: 0.300
- Mean recall: 0.736
- Mean F1-score: 0.426

## Holdout test set

- Accuracy: 0.694
- Balanced accuracy: 0.721
- Precision: 0.296
- Recall: 0.759
- F1-score: 0.426
- ROC-AUC: 0.784
- Average precision: 0.377

Confusion matrix in the recorded run, formatted as `[[TN, FP], [FN, TP]]`:

```text
[[5808, 2696],
 [ 360, 1136]]
```

## Interpretation

The target is imbalanced, so accuracy alone is insufficient. The selected threshold yields higher recall than precision; many flagged customers do not complete a booking. Use a pilot campaign and tune the decision threshold before operational use. Feature importance reports predictive association and does not establish causation.
