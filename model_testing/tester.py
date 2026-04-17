import torch
import random
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, DebertaV2ForSequenceClassification
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from textattack.attack_recipes import TextFoolerJin2019, DeepWordBugGao2018, BAEGarg2019
from textattack import Attack, Attacker, AttackArgs
from textattack.attack_recipes import AttackRecipe
from textattack.transformations import CompositeTransformation, WordSwapMaskedLM, WordSwapGradientBased, WordSwapEmbedding

from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as TA_Dataset

from textattack.transformations import WordSwapMaskedLM
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.attack_results import SuccessfulAttackResult

import os
import tensorflow_hub as hub

import logging
import tensorflow as tf


# Set a clean cache directory
os.environ["TFHUB_CACHE_DIR"] = "./tfhub_cache"

# Pre-download USE before running the attack
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("USE loaded successfully")

def prepare_attacks(dataloader, model, tokenizer, device, max_samples=50):
    model.eval()
    adversarial_data = []

    with torch.no_grad():
        for batch in dataloader:
            if len(adversarial_data) >= max_samples:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            b_input_ids = batch['input_ids']
            b_attn_mask = batch['attention_mask']
            b_labels = batch['labels']

            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            preds = torch.argmax(outputs.logits, dim=1).flatten()

            correct_mask = (preds == b_labels)
            
            input_ids = b_input_ids[correct_mask]
            labels = b_labels[correct_mask]

            for i in range(len(input_ids)):
                if len(adversarial_data) >= max_samples:
                    break
            
                token_list = input_ids[i].tolist()
                original_text = tokenizer.decode(token_list, skip_special_tokens=True)
                label = int(labels[i].item())
                adversarial_data.append((original_text, label))            

    return TA_Dataset(adversarial_data)

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

def main(args):

    models = {
        "roberta_base" : "../model_training/final_roberta_model",
        "deberta_base" : "../model_training/deberta-v3-hs-tuned",
        "deberta_adv" : "../model_training/deberta-v3-adversarial-final"
    }

    recipes = {
        "character" : DeepWordBugGao2018,
        "word": TextFoolerJin2019,
        "semantic": BAEGarg2019,
        "hybrid": DeBERTaAttack
    }

    # 1. Configuration
    TEST_DATA_PATH = "../data/test.csv" # Path to your small test file
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    # 2. Load Model and Tokenizer
    # Would to need to refactor to test all models against all recipes.
    if args.model_name == "roberta":
        MODEL_PATH = "../model_training/final_roberta_model"

        print(f"Loading model from {MODEL_PATH}...")

        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    elif "deberta" in args.model_name:
        if args.model_name == "deberta_adversarial_trained":
            MODEL_PATH = "../model_training/deberta-v3-adversarial-final"            
        else:
            MODEL_PATH = "../model_training/deberta-v3-hs-tuned"
        
        
        print(f"Loading model from {MODEL_PATH}...")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
        model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        print("Model doesn't not exist. Quitting...")
        return
    
    model.to(DEVICE)
    model.eval()

    # 3. Load and Prepare Test Data
    df = pd.read_csv(TEST_DATA_PATH)
    # Ensure columns match your training script names
    texts = df['Content'].tolist()
    labels = df['Label'].tolist()

    # Tokenization
    encodings = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )

    #Once all attacks work, then we optimize it to make sure it runs through all tests at once.
    if args.adversarial_attacks:
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        if args.adversarial_attacks == "character":
            attack = DeepWordBugGao2018.build(model_wrapper)
        elif args.adversarial_attacks == "word":
            attack = TextFoolerJin2019.build(model_wrapper)
        elif args.adversarial_attacks == "semantic":
            attack = BAEGarg2019.build(model_wrapper)
        elif args.adversarial_attacks == "hybrid":
            attack = DeBERTaAttack.build(model_wrapper)
        else:
            print("Invalid attack type. Quitting...")
            return    

    test_loader = DataLoader(TestDataset(encodings, labels), batch_size=8)

    if args.adversarial_attacks:
        attack_args = AttackArgs(
            log_to_csv=f'attack_results_{args.adversarial_attacks}.csv',
            num_examples=10,  
        )
        attack_dataset = prepare_attacks(test_loader, model, tokenizer, DEVICE)
        attacker = Attacker(attack, attack_dataset, attack_args)
        results = attacker.attack_dataset()

        successful_attacks = 0
        total_attempts = 0

        for result in results:
            total_attempts += 1
            if isinstance(result, SuccessfulAttackResult):
                successful_attacks += 1

        asr = (successful_attacks / total_attempts) * 100 if total_attempts > 0 else 0
        print(f"Attack Success Rate: {asr:.2f}%")    

    # # 4. Evaluation Loop
    # all_preds = []
    # all_labels = []

    # print("Running evaluation...")
    # with torch.no_grad():
    #     for batch in test_loader:
    #         input_ids = batch['input_ids'].to(DEVICE)
    #         attention_mask = batch['attention_mask'].to(DEVICE)
    #         targets = batch['labels'].to(DEVICE)
            
    #         outputs = model(input_ids, attention_mask=attention_mask)
    #         preds = torch.argmax(outputs.logits, dim=1)
            
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(targets.cpu().numpy())

    # # 5. Report Results
    # print("\n" + "="*30)
    # print("TEST PERFORMANCE REPORT")
    # print("="*30)
    # print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    # print("\nClassification Report:")
    # print(classification_report(all_labels, all_preds))

    # cm = confusion_matrix(all_labels, all_preds)

    # print("\nConfusion Matrix:")
    # print(cm)

    # filename = f"{args.model_name}_final_confusion_matrix.png"

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["hate", "non-hate"])
    # disp.plot()
    # disp.ax_.set_title(f"{args.model_name.capitalize()} - Final Confusion Matrix")
    # disp.figure_.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeBERTa Hate Speech Classifer')
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--adversarial_attacks', default=None)

    args = parser.parse_args()

    main(args)