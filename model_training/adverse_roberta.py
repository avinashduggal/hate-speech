import torch
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification

df = pd.read_csv('../data/train.csv')

# Char-level: Single key number typos
# Aug chance is higher since hate speech is more likely to use number typos to avoid filters
number_aug = nac.OcrAug(aug_char_p=0.2, aug_word_p=0.2)

# Single key character typos
typo_aug = nac.KeyboardAug(aug_char_p=0.1, aug_word_p=0.05)

# Word-level: Contextual Word Embeddings 
# Uses Wordnet to find synonyms to swap in
word_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
# Creates a seperate train set but with typos and avoidance techniques
def augment_batch(texts):
    augmented_texts = []
    counter = 0
    for text in texts:
        counter += 1
        original = text

        # Word level swaps for synonyms
        text = word_aug.augment(text)[0]

        # Apply character noise
        if torch.rand(1) > 0.5:
            text = number_aug.augment(text)[0]
        else:
            text = typo_aug.augment(text)[0]
        
        if counter % 100 == 0:
            print("\n-----------------------------------")
            print(f"Original: {original} \n Modifed: {text}")
        augmented_texts.append(text)
    return augmented_texts

def augment_batch_aggressive(texts):
    augmented_texts = []
    for text in texts:
        # Randomly choose between character typo or number type
        if torch.rand(1) > 0.5:
            text = number_aug.augment(text)[0]
        else:
            text = typo_aug.augment(text)[0]
        augmented_texts.append(text)
    return augmented_texts

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Content'].tolist(), df['Label'].tolist(), test_size=0.2, random_state=42
)

print("Modifying training data")
train_texts_hardened = augment_batch(train_texts)
# Combine original + modified for a larger, more robust dataset
train_texts_final = train_texts + train_texts_hardened
train_labels_final = train_labels + train_labels

# Tokenizing
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_data(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encodings, torch.tensor(labels)

train_encodings, train_y = tokenize_data(train_texts_final, train_labels_final)
val_encodings, val_y = tokenize_data(val_texts, val_labels)


train_encodings_clean, train_y_clean = tokenize_data(train_texts, train_labels)

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
train_loader_clean = DataLoader(ParagraphDataset(train_encodings_clean, train_y_clean), batch_size=16, shuffle=True)

val_loader = DataLoader(ParagraphDataset(val_encodings, val_y), batch_size=16)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)

# Freeze all -> then unfreeze last 7 layers
for param in model.parameters():
    param.requires_grad = False
for i in range(5, 12):
    for param in model.roberta.encoder.layer[i].parameters():
        param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# Training
model.train()

# 2 clean epochs
for epoch in range(2):
    total_loss = 0
    # Batch of 16
    batch_counter = 0
    for batch in train_loader_clean:
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
        if batch_counter % 10 == 0:
            print(f"Batch {batch_counter} | Clean Epoch {epoch} complete.")

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader_clean):.4f}")

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
        if batch_counter % 10 == 0:
            print(f"Batch {batch_counter} | Modified Epoch {epoch} complete.")

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
model.save_pretrained("./adverse_roberta")
tokenizer.save_pretrained("./adverse_roberta")