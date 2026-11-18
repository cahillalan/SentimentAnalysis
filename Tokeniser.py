from sacremoses import MosesTokenizer
from nltk.tokenize import word_tokenize
import pandas as pd

train_df = pd.read_csv("data/split/train.csv",keep_default_na=False)
test_df = pd.read_csv("data/split/test.csv",keep_default_na=False)
val_df = pd.read_csv("data/split/val.csv",keep_default_na=False)

print(train_df.head())

moses = MosesTokenizer(lang="en")

def standard_tokenize(text):
    return word_tokenize(text, preserve_line=True)

def tokenize(text):
    return moses.tokenize(text, return_str=False)

train_standard = train_df.copy()
train_moses = train_df.copy()

# Apply tokenisation
train_standard["text"] = train_standard["text"].apply(standard_tokenize)
train_moses["text"] = train_moses["text"].apply(tokenize)

test_standard = test_df.copy()
test_moses = test_df.copy()

# Apply tokenisation
test_standard["text"] = test_standard["text"].apply(standard_tokenize)
test_moses["text"] = test_moses["text"].apply(tokenize)

val_standard = val_df.copy()
val_moses = val_df.copy()

# Apply tokenisation
val_standard["text"] = val_standard["text"].apply(standard_tokenize)
val_moses["text"] = val_moses["text"].apply(tokenize)

train_standard.to_json("data/tokenized/train_standard.jsonl", orient="records", lines=True)
test_standard.to_json("data/tokenized/test_standard.jsonl", orient="records", lines=True)
val_standard.to_json("data/tokenized/val_standard.jsonl", orient="records", lines=True)

# Moses tokenisation
train_moses.to_json("data/tokenized/train_moses.jsonl", orient="records", lines=True)
test_moses.to_json("data/tokenized/test_moses.jsonl", orient="records", lines=True)
val_moses.to_json("data/tokenized/val_moses.jsonl", orient="records", lines=True)

test_df.to_csv("data/tokenized/test.csv")
train_df.to_csv("data/tokenized/train.csv")
val_df.to_csv("data/tokenized/val.csv")

