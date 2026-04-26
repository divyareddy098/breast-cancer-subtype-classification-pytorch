# Purpose: Load and preprocess TCGA RNA-seq data for model development

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# User inputs
expression_file = "data/tcga_expression.csv"   
metadata_file = "data/tcga_metadata.csv"       


# Load data
def load_data(expression_path, metadata_path):
    """
    Load expression and metadata files.
    """
    expression_df = pd.read_csv(expression_path, index_col=0)
    metadata_df = pd.read_csv(metadata_path)

    print("Expression data shape:", expression_df.shape)
    print("Metadata shape:", metadata_df.shape)

    return expression_df, metadata_df


# Preprocess data
def preprocess_data(expression_df, metadata_df, label_column="subtype"):
    """
    Align samples, encode subtype labels, and split into train/test sets.
    """
    if label_column not in metadata_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")

    # Align samples
    common_samples = expression_df.columns.intersection(metadata_df["sample_id"])
    expression_df = expression_df[common_samples]
    metadata_df = metadata_df[metadata_df["sample_id"].isin(common_samples)]

    # Transpose expression matrix so rows = samples
    X = expression_df.T.values
    y = metadata_df.set_index("sample_id").loc[common_samples, label_column].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Classes:", list(label_encoder.classes_))

    return X_train, X_test, y_train, y_test, label_encoder, scaler


# Main
if __name__ == "__main__":
    try:
        expression_df, metadata_df = load_data(expression_file, metadata_file)
        X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(
            expression_df,
            metadata_df,
            label_column="subtype"
        )

        print("Data preprocessing completed successfully.")
        print("Pipeline ready for model training.")

    except FileNotFoundError:
        print("Placeholder setup: input data files not found yet.")
        print("Expected files:")
        print(f" - {expression_file}")
        print(f" - {metadata_file}")
        print("Update file paths and rerun once TCGA data is available.")
