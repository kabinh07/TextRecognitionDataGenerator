import json
from tqdm import tqdm
import re
import pandas as pd
import string
import os
import random

def split_into_chunks(text, chunk_size=30):
    """
    Splits text into chunks at space, comma, or '।', keeping chunks <= chunk_size.
    Tries to keep chunk length close to chunk_size, but never exceeds.
    Never breaks a sentence mid-word; splits at the closest separator before exceeding chunk_size.
    """
    import re
    separators = [',', '।', ' ']
    chunks = []
    start = 0
    last_sep = -1

    for i, char in enumerate(text):
        if char in separators:
            last_sep = i
        if i - start + 1 > chunk_size:
            # If we have a separator before chunk_size, split there
            if last_sep >= start:
                chunk = text[start:last_sep+1].strip()
                if chunk:
                    chunks.append(chunk)
                start = last_sep + 1
            else:
                # No separator found, force split at chunk_size
                chunk = text[start:i].strip()
                if chunk:
                    chunks.append(chunk)
                start = i
            last_sep = -1
    # Add the last chunk
    if start < len(text):
        chunk = text[start:].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def clean_english_words(texts):
    new_texts = []
    for text in tqdm(texts, total=len(texts), desc="Cleaning english texts from bangla corpus"):
        # Remove English letters, numbers, and common exam patterns
        text = re.sub(r'[a-zA-Z0-9]+×[0-9]+=[0-9]+', '', text)  # Remove patterns like '5×1=5'
        text = re.sub(r'[a-zA-Z0-9]+', '', text)  # Remove remaining English/numbers
        # text = re.sub(r'[^\u0980-\u09FF\s]', '', text)  # Keep only Bangla chars and spaces
        text = ' '.join(text.split())  # Clean up whitespace
        
        # Only keep non-empty text that contains Bangla characters
        if text.strip() and re.search(r'[\u0980-\u09FF]', text):
            new_texts.append(text)
    
    return list(set(new_texts))

def clean_bangla_words(texts):
    new_texts = []
    for text in tqdm(texts, total=len(texts), desc="Cleaning bangla texts from english corpus"):
        # Remove Bangla characters
        text = re.sub(r'[\u0980-\u09FF]+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Only add non-empty strings that contain at least one letter/number
        if text.strip() and any(c.isalnum() for c in text):
            new_texts.append(text)
    
    return list(set(new_texts))


def create_bangla_list():
    with open(os.path.join(os.path.dirname(__file__), ('data/data_v2.json')), 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/train_labels.csv'), encoding='utf-8')
    texts = df['words'].to_list()
    texts = clean_english_words(texts)

    bangla_data = []
    for item in tqdm(data, total=len(data)):
        bangla_data.extend(split_into_chunks(item.get("author")))
        for chunk in item.get("published_date").split(", "):
            bangla_data.extend(split_into_chunks(chunk))
        for chunk in item.get("modification_date").split(", "):
            bangla_data.extend(split_into_chunks(chunk))
        bangla_data.extend(split_into_chunks(item.get("title")))
        bangla_data.extend(split_into_chunks(item.get("content")))

    bangla_data = clean_english_words(bangla_data)

    texts.extend(bangla_data)
    first_100k = texts[:100000]
    remaining = texts[100000:]
    random.shuffle(first_100k)
    random.shuffle(remaining)
    texts = first_100k + remaining

    with open("/app/list_data/bangla_list.json", 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)
    
def create_english_list():
    with open(os.path.join(os.path.dirname(__file__), ('data/english_news.json')), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/train_labels.csv'), encoding='utf-8')
    texts = df['words'].to_list()
    texts = clean_bangla_words(texts)

    english_data = []
    for item in tqdm(data, total=len(data)):
        english_data.extend(split_into_chunks(item))
    
    english_data = clean_bangla_words(english_data)
    
    texts.extend(english_data)
    first_100k = texts[:100000]
    remaining = texts[100000:]
    random.shuffle(first_100k)
    random.shuffle(remaining)
    texts = first_100k + remaining

    with open("/app/list_data/english_list.json", 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # create_bangla_list()
    # create_english_list()
    with open("/app/list_data/bangla_list.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(data[:20])