import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "data/processed/final_dataset.csv"
MODEL_FILE = "results/models/best_model.pt"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

RANDOM_SEED = 42
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 150

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BreastCancerSubtypeNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def main():
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["Sample", "Subtype"])
    y = df["Subtype"]

    gene_names = X.columns.tolist()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()

    X_values = X.values.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_values,
        y_encoded,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

    model = BreastCancerSubtypeNN(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    background_idx = np.random.choice(
        X_train.shape[0],
        size=min(BACKGROUND_SIZE, X_train.shape[0]),
        replace=False
    )

    explain_idx = np.random.choice(
        X_test.shape[0],
        size=min(EXPLAIN_SIZE, X_test.shape[0]),
        replace=False
    )

    background = torch.tensor(X_train[background_idx], dtype=torch.float32).to(DEVICE)
    explain_data = torch.tensor(X_test[explain_idx], dtype=torch.float32).to(DEVICE)

    print("Computing SHAP values...")
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(explain_data)

    X_explain_np = X_test[explain_idx]

    # SHAP sometimes returns list[class] or array[samples, genes, classes]
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values)
    else:
        shap_array = np.moveaxis(np.array(shap_values), -1, 0)

    # Overall gene importance across all classes
    overall_importance = np.mean(np.abs(shap_array), axis=(0, 1))

    top_overall = pd.DataFrame({
        "Gene": gene_names,
        "MeanAbsSHAP": overall_importance
    }).sort_values("MeanAbsSHAP", ascending=False)

    top_overall.to_csv("results/top_predictive_genes_overall.csv", index=False)

    # Top genes per subtype
    subtype_tables = []

    for class_idx, class_name in enumerate(class_names):
        class_importance = np.mean(np.abs(shap_array[class_idx]), axis=0)

        class_df = pd.DataFrame({
            "Subtype": class_name,
            "Gene": gene_names,
            "MeanAbsSHAP": class_importance
        }).sort_values("MeanAbsSHAP", ascending=False)

        class_df.head(50).to_csv(
            f"results/top_genes_{class_name}.csv",
            index=False
        )

        subtype_tables.append(class_df.head(50))

    pd.concat(subtype_tables).to_csv(
        "results/top_predictive_genes_by_subtype.csv",
        index=False
    )

    # SHAP summary plot for the strongest class by sample count
    main_class_idx = class_names.index("LumA") if "LumA" in class_names else 0

    shap.summary_plot(
        shap_array[main_class_idx],
        X_explain_np,
        feature_names=gene_names,
        show=False,
        max_display=25
    )

    plt.title(f"SHAP Summary Plot for {class_names[main_class_idx]}")
    plt.tight_layout()
    plt.savefig("figures/shap_summary_LumA.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Overall top genes bar plot
    top25 = top_overall.head(25).iloc[::-1]

    plt.figure(figsize=(8, 8))
    plt.barh(top25["Gene"], top25["MeanAbsSHAP"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Gene")
    plt.title("Top Predictive Genes Across Breast Cancer Subtypes")
    plt.tight_layout()
    plt.savefig("figures/top_predictive_genes_overall.png", dpi=300)
    plt.close()

    print("✅ SHAP interpretation complete")
    print("Saved:")
    print(" - results/top_predictive_genes_overall.csv")
    print(" - results/top_predictive_genes_by_subtype.csv")
    print(" - figures/shap_summary_LumA.png")
    print(" - figures/top_predictive_genes_overall.png")


if __name__ == "__main__":
    main()
