import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


# Config
DATA_FILE = "data/processed/final_dataset.csv"
MODEL_DIR = "results/models"
RESULTS_DIR = "results"

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


# Dataset
class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model
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


# Training / Evaluation
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs)
    }


def main():
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["Sample", "Subtype"])
    y = df["Subtype"]

    gene_names = X.columns.tolist()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_names = label_encoder.classes_.tolist()
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Input genes:", X.shape[1])
    print("Samples:", X.shape[0])

    X_values = X.values.astype(np.float32)

    # Train / validation / test split
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

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    train_dataset = GeneExpressionDataset(X_train, y_train)
    val_dataset = GeneExpressionDataset(X_val, y_val)
    test_dataset = GeneExpressionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Class imbalance handling
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    model = BreastCancerSubtypeNN(
        input_dim=X_train.shape[1],
        num_classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    history = []

    best_model_path = os.path.join(MODEL_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)

        scheduler.step(val_metrics["loss"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"]
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": X_train.shape[1],
                "num_classes": num_classes,
                "class_names": class_names,
                "gene_names": gene_names
            }, best_model_path)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion)

    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=class_names,
        digits=4
    )

    print("\nFinal Test Results")
    print("==================")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {test_metrics['f1_weighted']:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save report
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Save predictions
    pred_df = pd.DataFrame({
        "true_label": label_encoder.inverse_transform(test_metrics["labels"]),
        "predicted_label": label_encoder.inverse_transform(test_metrics["preds"])
    })

    for i, class_name in enumerate(class_names):
        pred_df[f"prob_{class_name}"] = test_metrics["probs"][:, i]

    pred_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

    # Save label mapping
    with open(os.path.join(RESULTS_DIR, "label_mapping.json"), "w") as f:
        json.dump({int(i): c for i, c in enumerate(class_names)}, f, indent=4)

    print("\n Training complete")
    print("Saved model:", best_model_path)
    print("Saved history: results/training_history.csv")
    print("Saved predictions: results/test_predictions.csv")
    print("Saved report: results/classification_report.txt")


if __name__ == "__main__":
    main()
