import torch
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from collections import Counter
import re

DATA_PATH = '../data/train.csv'
MODEL_SAVE_PATH = "./adverse_roberta_dynamic_2"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16
CLEAN_EPOCHS = 2
AUGMENTED_EPOCHS = 3
LEARNING_RATE = 2e-5

# Load and Split Data
df = pd.read_csv(DATA_PATH)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Content'].tolist(), df['Label'].tolist(), test_size=0.2, random_state=42
)

# Initialize Model & Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(DEVICE)

# Augments
number_aug = nac.OcrAug(aug_char_p=0.4, aug_word_p=1.0)
typo_aug = nac.KeyboardAug(aug_char_p=0.3, aug_word_p=1.0)
word_aug = naw.SynonymAug(aug_src='wordnet', aug_p=1.0)

# Get the words that correspond most to hate speech classification. Uses a scoring system
def get_hateful_impact_words(texts, labels, min_score=4.0):
    hate_words = Counter()
    neutral_words = Counter()
    for text, label in zip(texts, labels):
        words = re.findall(r'\w+', str(text).lower())
        if label == 1: hate_words.update(words)
        else: neutral_words.update(words)
            
    impact_keywords = set()
    for word, count in hate_words.items():
        score = count / (neutral_words[word] + 1)
        if count >= 5 and score >= min_score:
            impact_keywords.add(word)
    return impact_keywords

# Augmment the dataset with an even amount of typos, number typos, and synonym swaps
def augment_batch(texts, keywords):
    augmented_texts = []
    for text in texts:
        words = str(text).split()
        new_words = []
        for word in words:
            clean_word = word.lower().strip(".,!?:;")
            if clean_word in keywords:
                rand_val = torch.rand(1).item()
                try:
                    if rand_val < 0.33: target_word = word_aug.augment(word)[0]
                    elif rand_val < 0.66: target_word = number_aug.augment(word)[0]
                    else: target_word = typo_aug.augment(word)[0]
                    new_words.append(target_word)
                except: new_words.append(word)
            else: new_words.append(word)
        augmented_texts.append(" ".join(new_words))
    return augmented_texts

def tokenize_data(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encodings, torch.tensor(labels)

class ParagraphDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self): return len(self.labels)

# Layer Freezing Logic. Use 7 layers
# More layers are needed to see the differences in typos and augmentation
for param in model.parameters(): param.requires_grad = False
for i in range(5, 12):
    for param in model.roberta.encoder.layer[i].parameters(): param.requires_grad = True
for param in model.classifier.parameters(): param.requires_grad = True

optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Train on clean datasets first
print(f"Starting {CLEAN_EPOCHS} Clean Epochs...")

# Tokenize the clean text datasets
train_enc_clean, train_y_clean = tokenize_data(train_texts, train_labels)
clean_loader = DataLoader(ParagraphDataset(train_enc_clean, train_y_clean), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(CLEAN_EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(clean_loader):
        optim.zero_grad()
        outputs = model(batch['input_ids'].to(DEVICE), 
                        attention_mask=batch['attention_mask'].to(DEVICE), 
                        labels=batch['labels'].to(DEVICE))
        loss = outputs.loss
        loss.backward()
        optim.step()
        total_loss += loss.item()
    print(f"Clean Epoch {epoch+1} | Avg Loss: {total_loss/len(clean_loader):.4f}")

# Train on augmented datasets
print(f"\nStarting {AUGMENTED_EPOCHS} Dynamic Augmented Epochs...")

for epoch in range(AUGMENTED_EPOCHS):
    # Get new list of hate words for each epoch. Note: this doesn't seem to do anything
    current_keywords = get_hateful_impact_words(train_texts, train_labels)
    print(f"Augmented Epoch {epoch+1}: Identified {len(current_keywords)} high-impact words.")
    
    # Save each epoch's list of words to see if it changes.
    with open(f'bad_words_epoch_{epoch+1}.txt', 'w') as f:
        for word in sorted(current_keywords): f.write(f"{word}\n")

    # Redo Augmentation for each epoch
    print(f"Generating fresh augmented data for Epoch {epoch+1}...")
    train_hardened = augment_batch(train_texts, current_keywords)
    
    # Combine and Tokenize
    # We combine the augmented and the clean dataset so the model doesn't 'forget' the normal text
    epoch_texts = train_texts + train_hardened
    epoch_labels = train_labels + train_labels
    epoch_enc, epoch_y = tokenize_data(epoch_texts, epoch_labels)
    epoch_loader = DataLoader(ParagraphDataset(epoch_enc, epoch_y), batch_size=BATCH_SIZE, shuffle=True)

    # Train loop
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(epoch_loader):
        optim.zero_grad()
        outputs = model(batch['input_ids'].to(DEVICE), 
                        attention_mask=batch['attention_mask'].to(DEVICE), 
                        labels=batch['labels'].to(DEVICE))
        loss = outputs.loss
        loss.backward()
        optim.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(epoch_loader)} processed...")

    print(f"Augmented Epoch {epoch+1} complete | Avg Loss: {total_loss/len(epoch_loader):.4f}")

#Evaluation with a small validation set
model.eval()
val_enc, val_y = tokenize_data(val_texts, val_labels)
val_loader = DataLoader(ParagraphDataset(val_enc, val_y), batch_size=BATCH_SIZE)

predictions = []
with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

print("\nFinal Results (Validation Set):")
print(classification_report(val_labels, predictions))


# Save the model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)