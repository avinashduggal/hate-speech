import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/HateSpeechDatasetBalanced.csv")
print(f"Original dataset: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"Label distribution: 0={np.sum(df['Label'] == 0)}, 1={np.sum(df['Label'] == 1)}\n")

df = df.dropna(subset=["Content", "Label"])
df = df[df["Content"].str.strip() != ""]
df["Label"] = df["Label"].astype(int)
print(f"After dropping dirty rows: {len(df)} rows")

rng = np.random.default_rng(42)
hateful = df[df["Label"] == 1].index.to_numpy()
non_hateful = df[df["Label"] == 0].index.to_numpy()
hateful_sample = rng.choice(hateful, size=5000, replace=False)
non_hateful_sample = rng.choice(non_hateful, size=5000, replace=False)
sample_idx = np.concatenate([hateful_sample, non_hateful_sample])
rng.shuffle(sample_idx)

df = df.loc[sample_idx].reset_index(drop=True)
print(f"After subsampling: {len(df)} rows")
print(f"Label distribution: 0={np.sum(df['Label'] == 0)}, 1={np.sum(df['Label'] == 1)}\n")

train_df, temp_df = train_test_split(df, test_size=0.10, random_state=42, stratify=df["Label"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["Label"])

print(f"Train: {len(train_df)} rows (0={np.sum(train_df['Label'] == 0)}, 1={np.sum(train_df['Label'] == 1)})")
print(f"Val:   {len(val_df)} rows (0={np.sum(val_df['Label'] == 0)}, 1={np.sum(val_df['Label'] == 1)})")
print(f"Test:  {len(test_df)} rows (0={np.sum(test_df['Label'] == 0)}, 1={np.sum(test_df['Label'] == 1)})")

for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split_df[["Content", "Label"]].to_csv(f"../data/{name}.csv", index=False)

print("\nSaved train.csv, val.csv, test.csv to data/")