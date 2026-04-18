from transformers import AutoTokenizer, DebertaV2ForSequenceClassification
import numpy as np
import pandas as pd
import torch

from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as TADataset

from textattack import TrainingArgs
from textattack import Trainer as TATrainer

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.patch_utils import apply_sentence_encoder_patch
from utils.deberta_custom_recipe import DeBERTaAttack

apply_sentence_encoder_patch()

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

# Set to train for 3 epochs.
# Will attack 2% of Training data in 1 epoch to adversarially train the model.
training_args = TrainingArgs(
    output_dir="./deberta-v3-adversarial",
    num_epochs=3,
    num_clean_epochs=1,         
    attack_epoch_interval=2,
    num_train_adv_examples=0.02,     
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

# Save the model and tokenizer once finished.
model.save_pretrained("./deberta-v3-adversarial-final")
tokenizer.save_pretrained("./deberta-v3-adversarial-final")
print("Adversarial DeBERTa Model saved.")