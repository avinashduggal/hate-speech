from transformers import RobertaForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, classification_report
import numpy as np
import torch
from torch.utils.data import Dataset

# Switched to roberta-base
MODEL_PATH = "roberta-base"

"""
    Custom HateSpeechDataset Class to load in npz data.
"""
class HateSpeechDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.input_ids  = torch.tensor(data["input_ids"])
        self.attention_mask  = torch.tensor(data["attention_mask"])
        self.labels = torch.tensor(data["labels"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx]
        }

train_dataset = HateSpeechDataset("../data/train_roberta.npz")
val_dataset   = HateSpeechDataset("../data/val_roberta.npz")

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    id2label={0: "non-hate", 1: "hate"},
    label2id={"non-hate": 0, "hate": 1}
)

# RoBERTa uses the standard fast tokenizer by default
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

"""
    Computes the Accuracy and F1-Score.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

"""
    Prints the Classification Reports and save Confusion Matrices.
"""
def print_classification_report(dataset, ds_type):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    filename = f"RoBERTa_{ds_type.lower()}_confusion_matrix.png"
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-hate", "hate"])
    disp.plot(cmap="Blues") 
    disp.ax_.set_title(f"RoBERTa {ds_type} - Confusion Matrix")
    disp.figure_.savefig(filename, bbox_inches='tight')

    print(f"\n--- {ds_type} Classification Report ---")
    print(classification_report(labels, preds, target_names=["non-hate", "hate"]))

# Training Arguments
training_args = TrainingArguments(
    output_dir="./roberta-base-hs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,  
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  
    greater_is_better=True,
    fp16=False, # Standard fp16 can be unstable on MPS; keep False unless using CUDA
    dataloader_pin_memory=False,    
    logging_steps=200,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the model and tokenizer
model.save_pretrained("./roberta-base-hs-tuned")
tokenizer.save_pretrained("./roberta-base-hs-tuned")
print("RoBERTa Model saved.")

print_classification_report(train_dataset, "Train")
print_classification_report(val_dataset, "Val")