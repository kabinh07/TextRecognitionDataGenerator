import json
import collections
import sys
import re

def analyze_tokens(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Analyzing tokens in {file_path}...")
    
    token_counts = collections.Counter()
    class_token_counts = collections.defaultdict(collections.Counter)
    
    total_texts = 0
    
    for class_name, texts in data.items():
        for text in texts:
            if isinstance(text, str):
                # Simple whitespace splitting for now, stripping punctuation
                tokens = re.findall(r'\w+', text)
                token_counts.update(tokens)
                class_token_counts[class_name].update(tokens)
                total_texts += 1

    print(f"Total texts analyzed: {total_texts}")
    print("\n--- Top 20 Most Common Tokens (Overall) ---")
    for token, count in token_counts.most_common(20):
        percentage = (count / total_texts) * 100
        print(f"{token}: {count} ({percentage:.2f}%)")

    print("\n--- Class-wise High Frequency Tokens (Top 5 per class) ---")
    for class_name, counts in class_token_counts.items():
        print(f"\nClass: {class_name}")
        for token, count in counts.most_common(5):
            print(f"  {token}: {count}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_tokens(sys.argv[1])
    else:
        print("Usage: python analyze_tokens.py <json_file>")
