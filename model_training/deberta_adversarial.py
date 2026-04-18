from transformers import AutoTokenizer, DebertaV2ForSequenceClassification
import numpy as np
import pandas as pd
import torch

import transformers

from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as TADataset

from textattack import TrainingArgs
from textattack import Trainer as TATrainer
from textattack import Attacker, AttackArgs

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from textattack.attack_results import SuccessfulAttackResult

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.patch_utils import apply_sentence_encoder_patch
from utils.deberta_custom_recipe import DeBERTaAttack

apply_sentence_encoder_patch()

transformers.optimization.AdamW = torch.optim.AdamW

device = torch.device("mps" if torch.mps.is_available() else "cpu")

# Load base model, model wrapper, attack, and datasets.
MODEL_PATH = "./deberta-v3-hs-tuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, model_max_length=512)
model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)

wrapped_model = HuggingFaceModelWrapper(model, tokenizer)

attack = DeBERTaAttack.build(wrapped_model)

train_df = pd.read_csv("../data/train.csv")  
val_df   = pd.read_csv("../data/val.csv")

train_data = list(zip(train_df["Content"].tolist(), train_df["Label"].tolist()))
val_data   = list(zip(val_df["Content"].tolist(),   val_df["Label"].tolist()))

train_dataset = TADataset(train_data)
val_dataset   = TADataset(val_data)

"""
    Computes the Accuracy and F1-Score.
"""
def compute_metrics(model_wrapper, dataset):
    preds = []
    labels = []
    
    for text_input, label in dataset:
        output = model_wrapper(text_input['text'])
        preds.append(np.argmax(output))
        labels.append(label)
        
    return preds, labels

"""
    Prints the Classification Reports and save Confusion Matrices for the specfied subsets.
"""
def print_classification_report(model_wrapper, dataset, ds_type):
    preds, labels = compute_metrics(model_wrapper, dataset)

    filename = f"./results/deberta_adv_{ds_type.lower()}_confusion_matrix.png"
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["hate", "non-hate"])
    disp.plot()
    disp.ax_.set_title(f"DeBERTa {ds_type} - Confusion Matrix")
    disp.figure_.savefig(filename, bbox_inches='tight')

    print(classification_report(labels, preds, target_names=["non-hate", "hate"]))

"""
    Computes ASR with specified number of samples.
"""
def compute_ASR(attack, dataset, ds_type, num_examples):
        attack_args = AttackArgs(
            log_to_csv=f'./results/attack_{ds_type.lower()}_results.csv',
            num_examples=num_examples,
            disable_stdout=True, 
        )
        attacker = Attacker(attack, dataset, attack_args)
        results = attacker.attack_dataset()

        successful_attacks = 0
        total_attempts = 0

        for result in results:
            total_attempts += 1
            if isinstance(result, SuccessfulAttackResult):
                successful_attacks += 1

        asr = (successful_attacks / total_attempts) * 100 if total_attempts > 0 else 0
        print(f"Attack Success Rate: {asr:.2f}%")  


# Set to train for 3 epochs.
# Will attack 5% of training data in 1 epoch to adversarially train the model.
training_args = TrainingArgs(
    output_dir="./deberta-v3-adversarial",
    num_epochs=3,
    num_clean_epochs=1,         
    attack_epoch_interval=2,
    num_train_adv_examples=0.05,     
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,          
    num_warmup_steps=500,
    weight_decay=0.01,
    random_seed=42,
    checkpoint_interval_epochs=1,
    log_to_tb=False,            
)

# Monkey-patched input_columns to accept tuples.
train_dataset.input_columns = tuple(train_dataset.input_columns)

# Set TextAttack Trainer and train the model.
trainer = TATrainer(
    model_wrapper=wrapped_model,
    task_type="classification",
    attack=attack,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
)

trainer.train()

# Computes ASR
compute_ASR(attack, train_dataset, "Train", 10)
compute_ASR(attack, val_dataset, "Val", 5)

# Print classification reports and confusion matrices.
print_classification_report(wrapped_model, train_dataset, "Train")
print_classification_report(wrapped_model, val_dataset, "Val")

# Save the model and tokenizer once finished.
model.save_pretrained("./deberta-v3-adversarial-final")
tokenizer.save_pretrained("./deberta-v3-adversarial-final")
print("Adversarial DeBERTa Model saved.")