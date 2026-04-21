import pandas as pd
import torch
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import os

# Initialize Augmenters
AUG_MAP = {
    "leetspeak": nac.OcrAug(aug_char_p=0.6, aug_word_p=1.0, min_char=1),
    "semantic": naw.SynonymAug(aug_src='wordnet', aug_p=1.0),
    "typo": nac.KeyboardAug(aug_char_p=0.5, aug_word_p=1.0, min_char=1)
}

def load_keywords(file_path):
    try:
        with open(file_path, 'r') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return set()

def apply_specific_augmentation(text, label, keywords, augmenter_type):
    if pd.isna(text):
        return text
    
    # We only attempt to modify Label 1 (Hate)
    if label == 0:
        return text
    
    original_text = str(text)
    words = original_text.split()
    
    # Check if this row contains a keyword we can target
    has_keyword = any(word.lower().strip(".,!?:;") in keywords for word in words)
    if not has_keyword:
        return original_text # Keep original if no keywords found

    augmenter = AUG_MAP[augmenter_type]
    attempts = 0
    modified_text = original_text
    
    # Force modification through retries
    while modified_text == original_text and attempts < 5:
        attempts += 1
        new_words = []
        for word in words:
            clean_word = word.lower().strip(".,!?:;")
            if clean_word in keywords:
                try:
                    target_word = augmenter.augment(word)[0]
                    new_words.append(target_word)
                except:
                    new_words.append(word)
            else:
                new_words.append(word)
        modified_text = " ".join(new_words)
    
    return modified_text

def main():
    INPUT_FILE = '../data/test.csv'
    KEYWORDS_FILE = '../model_training/bad_words.txt'
    keywords = load_keywords(KEYWORDS_FILE)
    
    df_base = pd.read_csv(INPUT_FILE)
    print(f"Original test set loaded: {len(df_base)} rows.")

    for aug_name in AUG_MAP.keys():
        print(f"Generating '{aug_name}' test set...")
        
        # Copy original data to ensure fresh start for each file
        df_temp = df_base.copy()
        
        # Apply augmentation directly to the Content column
        df_temp['Content'] = df_temp.apply(
            lambda row: apply_specific_augmentation(row['Content'], row['Label'], keywords, aug_name), 
            axis=1
        )
        
        # Save every row, modified or not
        output_name = f"../data/test_{aug_name}.csv"
        df_temp.to_csv(output_name, index=False)
        
        # Comparison for logging
        changes = (df_base['Content'] != df_temp['Content']).sum()
        print(f"Saved {len(df_temp)} rows to {output_name}")
        print(f"Total modified rows in this set: {changes}")

if __name__ == "__main__":
    main()