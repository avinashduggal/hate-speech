import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df = pd.read_csv('../data/train.csv') 

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Content'].tolist(), 
    df['Label'].tolist(), 
    test_size=0.2, 
    random_state=42
)

# Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_data(texts, labels):
    encodings = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_tensors='pt'
    )
    return encodings, torch.tensor(labels)

train_encodings, train_y = tokenize_data(train_texts, train_labels)
val_encodings, val_y = tokenize_data(val_texts, val_labels)

class ParagraphDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ParagraphDataset(train_encodings, train_y)
val_dataset = ParagraphDataset(val_encodings, val_y)

device = "mps"
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optim = AdamW(model.parameters(), lr=2e-5)

# Freeze the entire RoBERTa base model initially
for param in model.roberta.parameters():
    param.requires_grad = False

# Unfreeze the last 2 layers and the classifier head
for i in range(10, 12): # Layers 10 and 11
    for param in model.roberta.encoder.layer[i].parameters():
        param.requires_grad = True


model.train()
for epoch in range(3): 
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")


save_path = "./final_roberta_model"

print(f"Saving final model to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

model.eval()
val_loader = DataLoader(val_dataset, batch_size=8)
predictions = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

print("\nFinal Results:")
print(classification_report(val_labels, predictions))