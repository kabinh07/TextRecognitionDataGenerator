import json
from tqdm import tqdm
import re
import pandas as pd
import string

with open("data_v2/data_v2.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

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
    for text in tqdm(texts, total=len(texts), desc="Cleaning english texts"):
        text = re.sub(r'[a-zA-Z0-9]+', '', text)
        if text.strip() != '' and not re.fullmatch(rf"[{re.escape(string.punctuation)}]+", text.strip()):
            new_texts.append(text)
            new_texts.append(text)
    return list(set(new_texts))

bangla_data = []
for item in tqdm(data, total=len(data)):
    bangla_data.extend(split_into_chunks(item.get("author")))
    for chunk in item.get("published_date").split(", "):
        bangla_data.extend(split_into_chunks(chunk))
    for chunk in item.get("modification_date").split(", "):
        bangla_data.extend(split_into_chunks(chunk))
    bangla_data.extend(split_into_chunks(item.get("title")))
    bangla_data.extend(split_into_chunks(item.get("content")))

df = pd.read_csv("data_v2/train_labels.csv", encoding='utf-8')
texts = df['words'].to_list()
texts = clean_english_words(texts)
bangla_data.extend(texts)

with open("data_v2/bangla_data.json", 'w', encoding='utf-8') as f:
    json.dump(bangla_data, f, ensure_ascii=False, indent=4)