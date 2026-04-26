import pandas as pd

# Load expression data
expr = pd.read_csv("data/processed/expression_processed.csv")

# Load clinical data
pheno = pd.read_csv("data/raw/BRCA_clinicalMatrix", sep="\t", low_memory=False)

# Keep only required columns
pheno = pheno[["sampleID", "PAM50Call_RNAseq"]]

# Rename columns
pheno.rename(columns={
    "sampleID": "Sample",
    "PAM50Call_RNAseq": "Subtype"
}, inplace=True)

# Clean sample IDs (match preprocessing)
pheno["Sample"] = pheno["Sample"].str[:12]

# Merge
df = expr.merge(pheno, on="Sample")

# Remove missing subtype
df = df.dropna(subset=["Subtype"])

# Remove "Normal-like" if present (optional but recommended)
df = df[df["Subtype"] != "Normal-like"]

# Save final dataset
df.to_csv("data/processed/final_dataset.csv", index=False)

print("Final dataset ready")
print("Shape:", df.shape)
print("\nSubtype counts:")
print(df["Subtype"].value_counts())
