import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "../model_training/final_roberta_model"
TEST_DATA_PATH = "../data/test.csv" 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

df = pd.read_csv(TEST_DATA_PATH)
texts = df['Content'].tolist()
labels = df['Label'].tolist()

encodings = tokenizer(
    texts, 
    padding=True, 
    truncation=True, 
    max_length=512, 
    return_tensors='pt'
)

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

test_loader = DataLoader(TestDataset(encodings, labels), batch_size=8)

all_preds = []
all_labels = []

print("Running evaluation...")
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        targets = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

print("\n" + "="*30)
print("TEST PERFORMANCE REPORT")
print("="*30)
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))