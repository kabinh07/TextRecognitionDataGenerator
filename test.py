from trdg.generators import GeneratorFromStrings
import json
import random
import os
from tqdm import tqdm

# generator =  GeneratorFromDict(
#     count=10,
#     language='bn'
# )

# for img, lbl in generator:
#     img.save(f'/app/out/{lbl}.png')

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

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), ('data_v2/data_small.json')), 'r') as f:
        json_data = json.load(f)
    # with open(os.path.join(os.path.dirname(__file__), ('data_v2/data_small.json')), 'w') as f:
    #     json.dump(json_data[:50], f)
    print(f"Loaded {len(json_data)} items from data.json")
    texts = ["মোঃ কাবিন হাসান কাঞ্চন", "বাসা নং-১৭৬, রোড নংঃ ৬, ব্লকঃ সি, বসুন্ধরা আবাসিক এলাকা, ঢাকা।"]
    # for item in tqdm(json_data, total=len(json_data), desc="Processing items"):
    #     text = item['content']
    #     transformed_text = split_paragraph_randomly(text, min_words=1, max_words=30, line_break_chance=0.2)
    #     cleaned_text = [item for item in transformed_text if item.strip() != '']
    #     texts.extend(cleaned_text)
    
    text_count = 10

    generator =  GeneratorFromStrings(
        strings=texts,
        count=text_count,
        size=32,
        language='bn',
        skewing_angle=2,
        random_skew=True,
        blur=0.5,
        random_blur=True,
        fit=True,
        word_split=True,
        # margins=(10,10,10,10),
        orientation=1
    )
    count = 0
    for img, lbl in tqdm(generator, total=text_count, desc="Generating images"):
        try:
            if not os.path.exists('/app/out/images'):
                os.makedirs('/app/out/images')
            if not os.path.exists('/app/out/labels'):
                os.makedirs('/app/out/labels')
            img.save(f'/app/out/images/{count}.png')
            with open(f'/app/out/labels/{count}.txt', 'w', encoding='utf-8') as f:
                f.write(lbl)
        except Exception as e:
            print(f"Error saving image {count}: {e}")
            continue
        count += 1
    # full_text = '\n'.join(texts)
    # with open(os.path.join(os.path.dirname(__file__), ('text_files/data_v2.txt')), 'w', encoding='utf-8') as f:
    #     f.write(full_text)
    # print(f"Saved transformed text to data_v2.txt with {len(texts)} lines")