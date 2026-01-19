import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_CSV = "train.csv"

# ======================
# LOAD DATA
# ======================
train_df = pd.read_csv(TRAIN_CSV, encoding="latin1")

# ======================
# Basic info
# ======================
print("Train set shape:", train_df.shape)
print("Columns:", train_df.columns)
print(train_df.head())

# ======================
# Tokenization functions
# ======================
def word_tokenize(text):
    return str(text).lower().split()

def char_tokenize(text):
    return list(str(text).lower())

train_df['search_tokens'] = train_df['search_term'].apply(word_tokenize)
train_df['title_tokens'] = train_df['product_title'].apply(word_tokenize)

# ======================
# Token counts
# ======================
train_df['search_len'] = train_df['search_tokens'].apply(len)
train_df['title_len'] = train_df['title_tokens'].apply(len)

print("\n=== Search Term Stats ===")
print(train_df['search_len'].describe())
print("Max search term length:", train_df['search_len'].max())

print("\n=== Product Title Stats ===")
print(train_df['title_len'].describe())
print("Max product title length:", train_df['title_len'].max())

# ======================
# Histogram of lengths
# ======================
plt.figure(figsize=(12,5))
plt.hist(train_df['search_len'], bins=20, alpha=0.7, label="Search term length")
plt.hist(train_df['title_len'], bins=50, alpha=0.7, label="Title length")
plt.xlabel("Token count")
plt.ylabel("Number of samples")
plt.title("Distribution of token lengths")
plt.legend()
plt.show()

# ======================
# Unique tokens
# ======================
all_search_tokens = [token for tokens in train_df['search_tokens'] for token in tokens]
all_title_tokens = [token for tokens in train_df['title_tokens'] for token in tokens]

print("Unique search tokens:", len(set(all_search_tokens)))
print("Unique title tokens:", len(set(all_title_tokens)))
from collections import Counter

title_token_counts = Counter([token for tokens in train_df['title_tokens'] for token in tokens])
rare_titles = sum(1 for t,c in title_token_counts.items() if c < 5)
print("Number of rare title words (<5 occurrences):", rare_titles)
# ======================
# CHAR VOCAB
# ======================
def build_char_vocab(dfs):
    chars = set()
    for df in dfs:
        for col in ["search_term", "product_title"]:
            for text in df[col].astype(str):
                chars.update(text.lower())
    char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    char2idx["<PAD>"] = 0
    return char2idx

char2idx = build_char_vocab([train_df])
print("\nUnique characters (char-level vocab):", len(char2idx) - 1)  # minus PAD

# Count unique characters in search terms and titles separately
search_chars = set("".join(train_df["search_term"].astype(str).str.lower()))
title_chars = set("".join(train_df["product_title"].astype(str).str.lower()))

print("Unique characters in search terms:", len(search_chars))
print("Unique characters in product titles:", len(title_chars))

# ======================
# Target distribution
# ======================
plt.figure(figsize=(8,4))
plt.hist(train_df['relevance'], bins=20, alpha=0.7, color='green')
plt.xlabel("Relevance")
plt.ylabel("Number of samples")
plt.title("Distribution of relevance scores")
plt.show()

# ======================
# Summary stats
# ======================
print("\n=== Relevance stats ===")
print(train_df['relevance'].describe())
