import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

preds = pd.read_csv("results/test_predictions.csv")

with open("results/label_mapping.json", "r") as f:
    label_mapping = json.load(f)

class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]

y_true = preds["true_label"]
y_pred = preds["predicted_label"]

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred, labels=class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted subtype")
plt.ylabel("True subtype")
plt.title("Confusion Matrix: Breast Cancer Subtype Classification")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.close()

# =========================
# ROC Curve
# =========================
y_true_bin = label_binarize(y_true, classes=class_names)

plt.figure(figsize=(8, 6))

auc_scores = {}

for i, class_name in enumerate(class_names):
    prob_col = f"prob_{class_name}"

    fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[prob_col])
    roc_auc = auc(fpr, tpr)
    auc_scores[class_name] = roc_auc

    plt.plot(fpr, tpr, label=f"{class_name} AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=300)
plt.close()

pd.DataFrame({
    "Subtype": list(auc_scores.keys()),
    "ROC_AUC": list(auc_scores.values())
}).to_csv("results/roc_auc_scores.csv", index=False)

# =========================
# Training Curves
# =========================
history = pd.read_csv("results/training_history.csv")

plt.figure(figsize=(8, 6))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("figures/training_loss_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
plt.plot(history["epoch"], history["val_f1_macro"], label="Validation Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Accuracy and Macro-F1")
plt.legend()
plt.tight_layout()
plt.savefig("figures/validation_metrics_curve.png", dpi=300)
plt.close()

print("Evaluation complete")
print("Saved figures:")
print(" - figures/confusion_matrix.png")
print(" - figures/roc_curve.png")
print(" - figures/training_loss_curve.png")
print(" - figures/validation_metrics_curve.png")
print("Saved ROC-AUC scores:")
print(" - results/roc_auc_scores.csv")
