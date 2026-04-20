import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# This file evaluates a model against test sets
MODEL_PATH = "../model_training/adverse_roberta_dynamic_2"
# List of files to test against
# Each of these test sets has a specific modification done to them
TEST_FILES = ["../data/test.csv", "../data/test_semantic.csv", "../data/test_leetspeak.csv", "../data/test_typo.csv"]
RESULTS_FILE = "adverse_roberta_2_evaluation_results.txt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Custom Dataset Class
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Open file for writing results
with open(RESULTS_FILE, "w") as f_out:
    f_out.write("ROBERTA EVALUATION REPORT\n")
    f_out.write("="*30 + "\n")

    for test_path in TEST_FILES:
        if not os.path.exists(test_path):
            print(f"Skipping {test_path}: File not found.")
            continue

        print(f"\nEvaluating: {test_path}")
        df = pd.read_csv(test_path)
        texts = df['Content'].astype(str).tolist()
        labels = df['Label'].tolist()

        encodings = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        )

        test_loader = DataLoader(TestDataset(encodings, labels), batch_size=8)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['labels'].to(DEVICE)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        # Formatting results
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)
        matrix = confusion_matrix(all_labels, all_preds)

        # Print to console
        print(f"Accuracy on {test_path}: {acc:.4f}")

        # Write to file
        f_out.write(f"\nTEST DATASET: {test_path}\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write("\nClassification Report:\n")
        f_out.write(report + "\n")
        f_out.write("\nConfusion Matrix:\n")
        f_out.write(str(matrix) + "\n")
        f_out.write("-" * 30 + "\n")

print(f"\nEvaluation complete. Results saved to {RESULTS_FILE}")