import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
MODEL_PATH = "../model_training/base_roberta"
# TEST_FILES = ["../data/test.csv", "../data/test_semantic.csv", "../data/test_leetspeak.csv", "../data/test_typo.csv"]
TEST_FILES = ["../data/test.csv"]

RESULTS_FILE = "base_roberta_evaluation_results.txt"
# Directory to save the PNGs
IMAGE_OUTPUT_DIR = "./results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

print(f"Loading model from {MODEL_PATH}...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

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

def save_confusion_matrix_png(matrix, test_name):
    """Generates and saves a heatmap for the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-hate', 'hate'], 
                yticklabels=['non-hate', 'hate'])
    
    plt.title(f"Roberta: Final Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Clean filename (e.g., ../data/test_typo.csv -> test_typo)
    clean_name = os.path.basename(test_name).replace('.csv', '')
    save_path = os.path.join(IMAGE_OUTPUT_DIR, f"cm_{clean_name}.png")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")

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

        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)
        matrix = confusion_matrix(all_labels, all_preds)

        # 1. Output to Console
        print(f"Accuracy on {test_path}: {acc:.4f}")

        # 2. Output to PNG
        save_confusion_matrix_png(matrix, test_path)

        # 3. Write to .txt File
        f_out.write(f"\nTEST DATASET: {test_path}\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write("\nClassification Report:\n")
        f_out.write(report + "\n")
        f_out.write("\nConfusion Matrix:\n")
        f_out.write(str(matrix) + "\n")
        f_out.write("-" * 30 + "\n")

print(f"\nEvaluation complete. Results saved to {RESULTS_FILE} and plots saved to {IMAGE_OUTPUT_DIR}")