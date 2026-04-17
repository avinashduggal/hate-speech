import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import textattack
from textattack.transformations import CompositeTransformation, WordSwapMaskedLM, WordSwapGradientBased, WordSwapEmbedding

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as TA_Dataset

from textattack.transformations import WordSwapMaskedLM
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification

from textattack.attack_recipes import TextFoolerJin2019, DeepWordBugGao2018, BAEGarg2019
from textattack import Attack, Attacker, AttackArgs

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from textattack.attack_results import SuccessfulAttackResult

def prepare_attacks(dataloader, model, tokenizer, device, max_samples=50):
    model.eval()
    adversarial_data = []

    with torch.no_grad():
        for batch in dataloader:
            if len(adversarial_data) >= max_samples:
                break

            b_input_ids, b_attn_mask, b_labels = [t.to(device) for t in batch]

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

def load_npz_to_dataloader(filepath, batch_size, shuffle=True):
    data = np.load(filepath)
    
    input_ids = torch.tensor(data['input_ids'])
    attn_masks = torch.tensor(data['attention_mask'])
    labels = torch.tensor(data['labels'])
    
    dataset = TensorDataset(input_ids, attn_masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, dataloader, optimizer, epochs, device):
    model.train()
    total_loss = 0
    for epoch in tqdm(range(epochs)):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} complete. Training Loss: {total_loss / len(dataloader)}")

def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).flatten()

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, accuracy, f1, all_labels, all_preds

def get_correct_samples(df, model, tokenizer, n=50):
    correct_samples = []
    model.eval()
    
    for _, row in df.iterrows():
        if len(correct_samples) >= n:
            break
            
        text = str(row["Content"])
        label = int(row["Label"])
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("mps")
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        
        if pred == label:
            correct_samples.append((text, label))
            
    return correct_samples

class HighConfidenceUntargeted(UntargetedClassification):
    def __init__(self, model, confidence_threshold=0.75, **kwargs):
        super().__init__(model, **kwargs)
        self.confidence_threshold = confidence_threshold

    def _is_goal_complete(self, model_output, _):
        predicted_class = model_output.argmax()
        confidence = model_output[predicted_class]
        return (
            predicted_class != self.ground_truth_output
            and confidence >= self.confidence_threshold
        )


def main(args):
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model_name = "../model training/deberta-v3-hs-tuned"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    model.to(device)

    # -----
    if args.adversarial_attacks:
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        if args.adversarial_attacks == "character":
            attack = DeepWordBugGao2018.build(model_wrapper)
        elif args.adversarial_attacks == "word":
            attack = TextFoolerJin2019.build(model_wrapper)
        elif args.adversarial_attacks == "semantic":
            attack = BAEGarg2019.build(model_wrapper)
        elif args.adversarial_attacks == "hybrid":
            goal_function = HighConfidenceUntargeted(model_wrapper, confidence_threshold=0.75)

            transformation = CompositeTransformation([
                WordSwapMaskedLM(method="bert", masked_language_model="roberta-base", max_candidates=15),
                WordSwapEmbedding(max_candidates=10),                                  
                WordSwapGradientBased(max_candidates=1)                                
            ])

            constraints = [
                RepeatModification(),
                StopwordModification(),
                UniversalSentenceEncoder(threshold=0.7)
            ]

            search_method = GreedyWordSwapWIR(wir_method="gradient")

            attack = textattack.Attack(goal_function, constraints, transformation, search_method)            
        else:
            print("Invalid attack type. Quitting...")
            return

    train_loader = load_npz_to_dataloader('../data/train_deberta.npz', batch_size=16)
    test_loader = load_npz_to_dataloader('../data/test_deberta.npz', batch_size=16, shuffle=False)

    # optimizer = AdamW(model.parameters(), lr=2e-5)

    # num_epochs = 3
    # if args.adversarial_attacks:
    #     model_path = f"deberta_test_{args.adversarial_attacks}.pt"
    # else:
    #     model_path = f"deberta_test.pt"

    # train_model(model, train_loader, optimizer, num_epochs, device)

    # torch.save(model.state_dict(), model_path)
    # print("Final model saved to:", model_path)

    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)

    # train_loss, train_accuracy, train_f1, _, _ = evaluate_model(model, train_loader, device)

    # print("\nTraining Metrics:")
    # print("Training Loss:", round(train_loss, 4))
    # print("Training Accuracy:", round(train_accuracy, 4))
    # print("Training F1-score:", round(train_f1, 4))

    # test_loss, test_accuracy, test_f1, y_true, y_pred = evaluate_model(model, test_loader, device)

    # print("\nFinal Test Loss:", round(test_loss, 4))
    # print("Final Test Accuracy:", round(test_accuracy, 4))
    # print("Final Test F1-score:", round(test_f1, 4))

    # cm = confusion_matrix(y_true, y_pred)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()

    # plt.title("Confusion Matrix -- DeBERTa")
    # plt.show()

    # print("\nClassification Report: ")
    # print(classification_report(y_true, y_pred))

    # print(classification_report(y_true, y_pred, digits=4))

    if args.adversarial_attacks:
        attack_args = AttackArgs(
            log_to_csv=f'attack_results_{args.adversarial_attacks}.csv',
            num_examples=50,  
        )
        attack_dataset = prepare_attacks(test_loader, model, tokenizer, device)
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

    #     results_table = pd.DataFrame({
    #         "Model": ["DeBERTa"],
    #         "Accuracy": [test_accuracy],
    #         "F1-score": [test_f1],
    #         "Attack Success Rate": [asr]
    #     })
    # # else:
    #     results_table = pd.DataFrame({
    #         "Model": ["DeBERTa"],
    #         "Accuracy": [test_accuracy],
    #         "F1-score": [test_f1]
    #     })

    # if args.adversarial_attacks:
    #     results_filename = f"deberta_results_{args.adversarial_attacks}_table.csv"
    # else:
    #     results_filename = f"deberta_results_table.csv"        

    # results_table.to_csv(results_filename, index=False)
    # print(f"Saved {results_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeBERTa Hate Speech Classifer')
    parser.add_argument('--adversarial_attacks', default=None)

    args = parser.parse_args()

    main(args)