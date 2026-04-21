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
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encodings, torch.tensor(labels)

train_encodings, train_y = tokenize_data(train_texts, train_labels)
val_encodings, val_y = tokenize_data(val_texts, val_labels)

# Modify data for use in the training loop
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


train_loader = DataLoader(ParagraphDataset(train_encodings, train_y), batch_size=16, shuffle=True)
val_loader = DataLoader(ParagraphDataset(val_encodings, val_y), batch_size=16)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)


# Freeze all -> then unfreeze last 4 layers
for param in model.parameters():
    param.requires_grad = False
for i in range(8, 12):
    for param in model.roberta.encoder.layer[i].parameters():
        param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# Training
model.train()
# 3 Epochs to train over
for epoch in range(3):
    total_loss = 0
    # Batch of 16
    batch_counter = 0
    for batch in train_loader:
        batch_counter += 1
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backpropagation
        loss.backward()
        optim.step()
        # Keep track of loss so we can see progress of training
        total_loss += loss.item()
        print(f"Batch {batch_counter} | Epoch {epoch} complete.")
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")


# Evaluation
model.eval()
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


# Save the model
model.save_pretrained("./base_roberta")
tokenizer.save_pretrained("./base_roberta")