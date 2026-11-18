from sacremoses import MosesTokenizer
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


nltk.download("stopwords")
nltk.download("wordnet")


stop_words = set(stopwords.words('english'))
##couldnt get punctuation to work.

train_df = pd.read_csv("data/split/train.csv",keep_default_na=False)
test_df = pd.read_csv("data/split/test.csv",keep_default_na=False)
val_df = pd.read_csv("data/split/val.csv",keep_default_na=False)

print(train_df.head())

moses = MosesTokenizer(lang="en")

def standard_tokenize(text):
    return word_tokenize(text, preserve_line=True)

def tokenize(text):
    return moses.tokenize(text, return_str=False)

def remove_stopwords(tokens):
    return [
        word for word in tokens
        if word.lower() not in stop_words
    ]

## pos brought errors so removed it
def lematize_words(tokens_stopped):
    return[lemmatizer.lemmatize(word) for word in tokens_stopped]

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

# Apply Stop words
val_standard["text"] = val_standard["text"].apply(remove_stopwords)
val_moses["text"] = val_moses["text"].apply(remove_stopwords)
test_standard["text"] = test_standard["text"].apply(remove_stopwords)
test_moses["text"] = test_moses["text"].apply(remove_stopwords)
train_standard["text"] = train_standard["text"].apply(remove_stopwords)
train_moses["text"] = train_moses["text"].apply(remove_stopwords)

# Apply Lema
val_standard["text"] = val_standard["text"].apply(lematize_words)
val_moses["text"] = val_moses["text"].apply(lematize_words)
test_standard["text"] = test_standard["text"].apply(lematize_words)
test_moses["text"] = test_moses["text"].apply(lematize_words)
train_standard["text"] = train_standard["text"].apply(lematize_words)
train_moses["text"] = train_moses["text"].apply(lematize_words)


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

