from trdg.generators import GeneratorFromStrings
import json
import random
import os
from tqdm import tqdm
import pandas as pd
import re
import string
import argparse
import sys

def clean_english_words(texts):
    new_texts = []
    for text in tqdm(texts, total=len(texts), desc="Cleaning english texts"):
        text = re.sub(r'[a-zA-Z0-9]+', '', text)
        if text.strip() != '' and not re.fullmatch(rf"[{re.escape(string.punctuation)}]+", text.strip()):
            new_texts.append(text)
            new_texts.append(text)
    return new_texts

def clean_bangla_words(texts):
    new_texts = []
    for text in tqdm(texts, total=len(texts), desc="Cleaning english texts"):
        text = re.sub(r'[\u0980-\u09FF]+', '', text)
        if text.strip() != '' and not re.fullmatch(rf"[{re.escape(string.punctuation)}]+", text.strip()):
            new_texts.append(text)
            new_texts.append(text)
    return new_texts

def split_paragraph_randomly(paragraph, min_words=1, max_words=10, line_break_chance=0.2):
    """
    Splits a paragraph into lines with random lengths (between min_words and max_words).
    Randomly inserts paragraph breaks as well.

    Args:
        paragraph (str): The input paragraph.
        min_words (int): Minimum words per line.
        max_words (int): Maximum words per line.
        line_break_chance (float): Probability of inserting an extra line break after a line.

    Returns:
        str: The transformed paragraph with random line splits and breaks.
    """
    words = paragraph.split()
    lines = []
    i = 0

    while i < len(words):
        line_len = random.randint(min_words, max_words)
        line_words = words[i:i + line_len]
        i += line_len
        line = ' '.join(line_words)
        if line.strip() == '':
            continue
        lines.append(line)

        # Randomly add extra line break (simulate paragraph)
        if random.random() < line_break_chance:
            lines.append("\n")  # An empty string becomes a paragraph break on join

    return lines

# if __name__ == "__main__":
#     with open(os.path.join(os.path.dirname(__file__), ('data_v2/english_news.json')), 'r', encoding='utf-8') as f:
#         json_data = json.load(f)
#     print(f"Loaded {len(json_data)} items from data.json")
#     random.shuffle(json_data)
#     json_data = json_data[:50000]

#     df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data_v2/train_labels.csv'), encoding='utf-8')
#     texts = df['words'].to_list()

#     print(f"Total texts from dataframe: {len(texts)}")
    
#     orietation = 0
#     text_count = 50000
#     font_size = 20
#     blur = 1.5
#     lanuage = 'en'

#     for item in tqdm(json_data, total=len(json_data), desc="Processing items"):
#         text = ""
#         if isinstance(item, dict):
#             text = item['content']
#         elif isinstance(item, str):
#             text = item
#         else:
#             continue
#         if orietation == 0:
#             transformed_text = split_paragraph_randomly(text, min_words=1, max_words=10, line_break_chance=0.0)
#             font_size = 32
#             blur = 0.1
#         else:
#             transformed_text = split_paragraph_randomly(text, min_words=1, max_words=30, line_break_chance=0.2)
#         cleaned_text = [item for item in transformed_text if item.strip() != '']
#         texts.extend(cleaned_text)
    
#     if lanuage == 'bn':
#         texts = clean_english_words(texts)
#     if lanuage == 'en':
#         texts = clean_bangla_words(texts)
#     texts = list(set(texts))
#     random.shuffle(texts)

#     print("\n\nParameters:\nOrientation:", orietation, "\nText Count:", len(texts), "\nFont Size:", font_size, "\nBlur:", blur, "\n\n")

#     generator =  GeneratorFromStrings(
#         strings=texts,
#         count=text_count,
#         size=font_size,
#         language=lanuage,
#         skewing_angle=3,
#         random_skew=True,
#         distorsion_type=0,
#         blur=1.5,
#         random_blur=True,
#         fit=True,
#         word_split=True,
#         background_type=3,
#         orientation=orietation
#     )
#     count = 50000
#     for img, lbl in tqdm(generator, total=text_count, desc="Generating images"):
#         try:
#             if not os.path.exists('/app/out/images'):
#                 os.makedirs('/app/out/images')
#             if not os.path.exists('/app/out/labels'):
#                 os.makedirs('/app/out/labels')
#             img.save(f'/app/out/images/{count}.png')
#             with open(f'/app/out/labels/{count}.txt', 'w', encoding='utf-8') as f:
#                 f.write(lbl)
#         except Exception as e:
#             print(f"Error saving image {count}: {e}")
#             continue
#         count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic text images.")
    parser.add_argument('--language', type=str, choices=['en', 'bn'], default='bn', help='Language of the text (en for English, bn for Bangla)')
    parser.add_argument('--text_count', type=int, default=100, help='Number of text images to generate')
    parser.add_argument('--orientation', type=int, choices=[0, 1], default=0, help='Orientation of the text (0: horizontal, 1: vertical)')
    parser.add_argument('--font_size', type=int, default=20, help='Font size of the text')
    parser.add_argument('--blur', type=float, default=1.5, help='Blur level of the text images')
    parser.add_argument('--use_list', action='store_true', help='Use list data for text generation')
    args = parser.parse_args()

    language = args.language
    text_count = args.text_count
    orietation = args.orientation
    font_size = args.font_size
    blur = args.blur
    use_list = args.use_list

    print("\n\nParameters:\nOrientation:", orietation, "\nFont Size:", font_size, "\nBlur:", blur, "\n\nUse_list:", use_list, "\n\n")

    # Loading JSON corpuses from files
    if language == 'en' and not use_list:
        with open(os.path.join(os.path.dirname(__file__), ('data/english_news.json')), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    elif language == 'bn' and not use_list:
        with open(os.path.join(os.path.dirname(__file__), ('data/data_V2.json')), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    elif language == 'bn' and use_list:
        with open(os.path.join(os.path.dirname(__file__), ('list_data/bangla_list.json')), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    elif language == 'en' and use_list:
        with open(os.path.join(os.path.dirname(__file__), ('data/english_data.json')), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        print("Invalid language choice. Please choose 'en' or 'bn'.")
        sys.exit(1)

    print(f"Loaded {len(json_data)} items")

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/train_labels.csv'), encoding='utf-8')
    texts = df['words'].to_list()

    print(f"Total texts from dataframe: {len(texts)}")
    json_data = json_data[:text_count]
    for item in tqdm(json_data, total=len(json_data), desc="Processing items"):
        text = ""
        if isinstance(item, dict):
            text = item['content']
        elif isinstance(item, str):
            text = item
        else:
            continue
        if orietation == 0:
            if use_list:
                transformed_text = json_data
                font_size = 32
                blur = 0.1
            else:
                transformed_text = split_paragraph_randomly(text, min_words=1, max_words=10, line_break_chance=0.0)
                font_size = 32
                blur = 0.1
        else:
            if use_list:
                transformed_text = json_data
            else:
                transformed_text = split_paragraph_randomly(text, min_words=1, max_words=30, line_break_chance=0.2)
        cleaned_text = [item for item in transformed_text if item.strip() != '']
        texts.extend(cleaned_text)
    
    if language == 'bn':
        texts = clean_english_words(texts)
    if language == 'en':
        texts = clean_bangla_words(texts)
    texts = list(set(texts))
    random.shuffle(texts)

    generator =  GeneratorFromStrings(
        strings=texts,
        count=text_count,
        size=font_size,
        language=language,
        skewing_angle=3,
        random_skew=True,
        distorsion_type=0,
        blur=1.5,
        random_blur=True,
        fit=True,
        word_split=True,
        background_type=3,
        orientation=orietation
    )
    count = 0
    if language == 'bn':
        output_dir = '/app/output/bangla'
    else:
        output_dir = '/app/output/english'
    for img, lbl in tqdm(generator, total=text_count, desc="Generating images"):
        try:
            image_dir = os.path.join(output_dir, 'images')
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            label_dir = os.path.join(output_dir, 'labels')
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            img.save(os.path.join(image_dir, f'{language}_{count}.png'))
            with open(os.path.join(label_dir, f'{language}_{count}.txt'), 'w', encoding='utf-8') as f:
                f.write(lbl)
        except Exception as e:
            print(f"Error saving image {count}: {e}")
            continue
        count += 1