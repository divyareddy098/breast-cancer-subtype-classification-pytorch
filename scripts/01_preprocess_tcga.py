import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FILE = "data/raw/HiSeqV2"
OUT_FILE = "data/processed/expression_processed.csv"
TOP_GENE_FILE = "results/top_variable_genes.csv"

MIN_NONZERO_SAMPLES = 50
TOP_N_GENES = 5000

def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(RAW_FILE):
        print(f"❌ File not found: {RAW_FILE}")
        print("Check files using: ls data/raw")
        return

    print("Loading expression data...")
    expr = pd.read_csv(RAW_FILE, sep="\t")

    expr.rename(columns={expr.columns[0]: "Gene"}, inplace=True)

    # Clean gene names such as TP53|7157 or ?|100130426
    expr["Gene"] = expr["Gene"].astype(str).str.split("|", regex=False).str[0]
    expr = expr[expr["Gene"] != "?"]

    # Average duplicated gene symbols
    expr = expr.groupby("Gene").mean(numeric_only=True)

    print("Raw gene x sample matrix:", expr.shape)

    # Transpose: samples as rows, genes as columns
    expr = expr.T
    expr.index.name = "Sample"

    # Clean TCGA barcode to patient-level ID
    expr.reset_index(inplace=True)
    expr["Sample"] = expr["Sample"].astype(str).str[:12]

    # If multiple tumor samples map to same patient, average them
    expr = expr.groupby("Sample").mean(numeric_only=True)

    print("Sample x gene matrix:", expr.shape)

    # Remove genes expressed in too few samples
    nonzero_counts = (expr != 0).sum(axis=0)
    expr = expr.loc[:, nonzero_counts >= MIN_NONZERO_SAMPLES]

    print("After low-expression filtering:", expr.shape)

    # Select top variable genes
    variances = expr.var(axis=0).sort_values(ascending=False)
    top_genes = variances.head(TOP_N_GENES).index.tolist()

    pd.DataFrame({
        "Gene": top_genes,
        "Variance": variances.loc[top_genes].values
    }).to_csv(TOP_GENE_FILE, index=False)

    expr = expr[top_genes]

    print("After top variable gene selection:", expr.shape)

    # Standardize expression values for neural network training
    scaler = StandardScaler()
    expr_scaled = pd.DataFrame(
        scaler.fit_transform(expr),
        index=expr.index,
        columns=expr.columns
    )

    expr_scaled.reset_index(inplace=True)
    expr_scaled.to_csv(OUT_FILE, index=False)

    print(" Advanced preprocessing complete")
    print("Processed expression saved to:", OUT_FILE)
    print("Top variable genes saved to:", TOP_GENE_FILE)
    print("Final shape:", expr_scaled.shape)

if __name__ == "__main__":
    main()
