import numpy as np
import pandas as pd
import sentencepiece
from transformers import RobertaTokenizer, AutoTokenizer
import argparse

def main(args):

    model_name = ""

    if args.tokenizer == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.tokenizer == "deberta":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    else:
        if args.model_name:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        else:
            print("Please specify a model!")
            return

    for name in ["train", "val", "test"]:
        df = pd.read_csv(f"../data/{name}.csv")

        encodings = tokenizer(
            df["Content"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
            return_token_type_ids=False
        )

        if args.model_name:
            filename = f"../data/{name}_{args.model_name}.npz"
        else:
            filename = f"../data/{name}_{args.tokenizer}.npz"
        
        np.savez(
            filename,
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            labels=df["Label"].values.astype(np.int64),
        )

        print(f"{name}: input_ids shape {encodings['input_ids'].shape}")

    if args.model_name:
        print(f"\nSaved train_{args.model_name}.npz, val_{args.model_name}.npz, test_{args.model_name}.npz to data/")
    else:
        print(f"\nSaved train_{args.tokenizer}.npz, val_{args.tokenizer}.npz, test_{args.tokenizer}.npz to data/")       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT_Tokenizer')
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--model_name', default=None)

    args = parser.parse_args()

    main(args)