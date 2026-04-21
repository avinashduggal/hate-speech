import numpy as np
import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for name in ["train", "val", "test"]:
    df = pd.read_csv(f"../data/{name}.csv")

    encodings = tokenizer(
        df["Content"].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="np",
    )

    np.savez(
        f"../data/{name}.npz",
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        labels=df["Label"].values.astype(np.int64),
    )

    print(f"{name}: input_ids shape {encodings['input_ids'].shape}")

print("\nSaved train.npz, val.npz, test.npz to data/")