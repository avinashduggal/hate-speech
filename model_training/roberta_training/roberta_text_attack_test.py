import os
# Force TensorFlow to skip the GPU/Metal entirely
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# Prevent macOS multiprocessing deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Optional: Ensure TextAttack doesn't spin up too many worker threads on Mac
os.environ["TEXTATTACK_NUM_WORKERS"] = "0"

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

print("--- System environment configured ---")
from transformers import RobertaTokenizer, RobertaForSequenceClassification

print("---Imported tranfsormers---")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Sys path appended")
from utils.patch_utils import apply_sentence_encoder_patch
from utils.custom_recipe import DeBERTaAttack
print("Imported utils")
apply_sentence_encoder_patch()
print("applied encoder")
transformers.optimization.AdamW = torch.optim.AdamW
print("AdamW optimized")
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Device chosen")
# Load base model, model wrapper, attack, and datasets.
MODEL_PATH = "../roberta_models/base_roberta"

print("--- Initializing Model and Tokenizer ---")

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(device)
model.to(device)

print("--- Wrapping Model ---")
wrapped_model = HuggingFaceModelWrapper(model, tokenizer)

print("--- Building Attack Recipe (This may take a minute) ---")
attack = DeBERTaAttack.build(wrapped_model)

print("--- Loading Datasets ---")
train_df = pd.read_csv("../../data/small.csv")  
val_df   = pd.read_csv("../../data/val.csv")

train_data = list(zip(train_df["Content"].tolist(), train_df["Label"].tolist()))
val_data   = list(zip(val_df["Content"].tolist(),   val_df["Label"].tolist()))

train_dataset = TADataset(train_data)
val_dataset   = TADataset(val_data)

"""
    Computes the Accuracy and F1-Score.
"""
def compute_metrics(model_wrapper, datasetnum_train_adv_examples = int(len(train_df) * 0.10)):
    preds = []
    labels = []
    with torch.no_grad():   
        for text_input, label in dataset:
            output = model_wrapper(text_input['text'])
            preds.append(np.argmax(output))
            labels.append(label)
        
    return preds, labels

"""
    Prints the Classification Reports and save Confusion Matrices for the specfied subsets.
"""
def print_classification_report(model_wrapper, dataset, ds_type):
    print(f"Computing metrics for {ds_type} set.")
    preds, labels = compute_metrics(model_wrapper, dataset)

    print(f"Generating Confusion Matrix for {ds_type} set.")
    filename = f"./results/roberta_adv_{ds_type.lower()}_confusion_matrix.png"
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["hate", "non-hate"])
    disp.plot()
    disp.ax_.set_title(f"RoBerta {ds_type} - Confusion Matrix")
    disp.figure_.savefig(filename, bbox_inches='tight')

    print(f"Loading Classification Report for {ds_type} set.")
    print(classification_report(labels, preds, target_names=["non-hate", "hate"]))
    print(f"Finished printing classification report and confusion matrix for {ds_type}!")

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
    output_dir="./roberta-adversarial",
    num_epochs=3,
    num_clean_epochs=1,         
    attack_epoch_interval=2,
    num_train_adv_examples = int(len(train_df) * 0.10),     
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
val_dataset.input_columns = tuple(val_dataset.input_columns)
# Set TextAttack Trainer and train the model.
print("--- Initializing Trainer ---")
trainer = TATrainer(
    model_wrapper=wrapped_model,
    task_type="classification",
    attack=attack,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
)

print("--- Starting Training Loop ---")
trainer.train()

# Save the model and tokenizer once finished.
model.save_pretrained("./roberta-adversarial-hybrid-final")
tokenizer.save_pretrained("./roberta-adversarial-hybrid-final")
print("Adversarial Roberta Model saved.")

model.eval()

# Computes ASR
compute_ASR(attack, train_dataset, "Train", 10)
compute_ASR(attack, val_dataset, "Val", 5)

# Print classification reports and confusion matrices.
print_classification_report(wrapped_model, train_dataset, "Train")
print_classification_report(wrapped_model, val_dataset, "Val")