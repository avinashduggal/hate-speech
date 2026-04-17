from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
import torch
import transformers

from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as TADataset
from textattack import Attack, TrainingArgs
from textattack import Trainer as TATrainer
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import CompositeTransformation, WordSwapMaskedLM, WordSwapGradientBased, WordSwapEmbedding
from textattack.attack_recipes import AttackRecipe

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, classification_report

transformers.optimization.AdamW = torch.optim.AdamW

class DeBERTaAttack(AttackRecipe):
    @staticmethod
    def build(model_wrapper): 
        goal_function = UntargetedClassification(model_wrapper)

        transformation = CompositeTransformation([
            WordSwapMaskedLM(method="bert-attack", masked_language_model="roberta-base", max_candidates=15),
            WordSwapEmbedding(max_candidates=10),                                                              
        ])

        constraints = [
            RepeatModification(),
            StopwordModification(),
            UniversalSentenceEncoder(threshold=0.7)
        ]

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)    


device = torch.device("mps" if torch.mps.is_available() else "cpu")

MODEL_PATH = "./deberta-v3-hatespeech-hs-tuned"

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

training_args = TrainingArgs(
    output_dir="./deberta-v3-adversarial",
    num_epochs=2,
    num_clean_epochs=1,         
    attack_epoch_interval=1,
    # attack_epoch_interval=2,
    num_train_adv_examples=0.1,     
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,          
    num_warmup_steps=500,
    weight_decay=0.01,
    random_seed=42,
    checkpoint_interval_epochs=1,
    log_to_tb=False,            
)

trainer = TATrainer(
    model_wrapper=wrapped_model,
    task_type="classification",
    attack=attack,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
)

trainer.train()

model.save_pretrained("./deberta-v3-adversarial-final")
tokenizer.save_pretrained("./deberta-v3-adversarial-final")
print("Adversarial DeBERTa Model saved.")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

eval_args = TrainingArguments(
    output_dir="./eval_output",
    per_device_train_batch_size=32,    
    per_device_eval_batch_size=32,
    fp16=torch.mps.is_available(),
    report_to="none"
)

hf_trainer = Trainer(
    model=model,
    args=eval_args,
    train_dataset=train_dataset,    
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

def print_classification_report(dataset, ds_type):
    predictions = hf_trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    filename = f"DeBERTa_{ds_type.lower()}_confusion_matrix.png"
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["hate", "non-hate"])
    disp.plot()
    disp.ax_.set_title(f"DeBERTa {ds_type} - Confusion Matrix")
    disp.figure_.savefig(filename, bbox_inches='tight')

    print(classification_report(labels, preds, target_names=["non-hate", "hate"]))

print_classification_report(train_dataset, "Train")
print_classification_report(val_dataset, "Val")
