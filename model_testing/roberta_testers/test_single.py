import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification



#This file allows for custom input and uses both models to give their predictions
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = "../model_training/base_roberta"

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()


model_path_2 = "../model_training/adverse_roberta_dynamic_2"

tokenizer_2 = RobertaTokenizer.from_pretrained(model_path_2)
model_2 = RobertaForSequenceClassification.from_pretrained(model_path_2).to(device)
model_2.eval()

def predict(text):
    # tokenize the input for processing
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=True
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    label = "Hate Speech" if prediction == 1 else "Neutral"
    return label, confidence

def predict_2(text):
    # tokenize the input for processing
    inputs = tokenizer_2(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=True
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model_2(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    label = "Hate Speech" if prediction == 1 else "Neutral"
    return label, confidence


print("Custom input mode: Type 'exit' to stop")
while True:
    user_input = input("Enter text to test: ")
    if user_input.lower() == 'exit':
        break
    result, score = predict(user_input)
    result_2, score_2 = predict_2(user_input)
    print(f"Base Prediction: {result} | Confidence: {score:.2%}\n")
    print(f"Adverse Prediction: {result_2} | Confidence: {score_2:.2%}\n")