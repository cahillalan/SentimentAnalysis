from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt




# Load tokenized data
train_moses = pd.read_json("data/tokenized/train_moses.jsonl", lines=True)
test_moses = pd.read_json("data/tokenized/test_moses.jsonl", lines=True)
val_moses = pd.read_json("data/tokenized/val_moses.jsonl", lines=True)

train_stan = pd.read_json("data/tokenized/train_standard.jsonl", lines=True)
test_stan = pd.read_json("data/tokenized/test_standard.jsonl", lines=True)
val_stan = pd.read_json("data/tokenized/val_standard.jsonl", lines=True)

# Separate TF-IDF vectorizers
tfidf_stan = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    lowercase=False
)

tfidf_moses = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    lowercase=False
)

# Training
xstanfit = tfidf_stan.fit_transform(train_stan["text"])
xmosesfit = tfidf_moses.fit_transform(train_moses["text"])

# Validation
xstanval = tfidf_stan.transform(val_stan["text"])
xmosesval = tfidf_moses.transform(val_moses["text"])

# Testing
xstantest = tfidf_stan.transform(test_stan["text"])
xmosestest = tfidf_moses.transform(test_moses["text"])

# Sentiment labels
y_train = train_stan["sentiment"]
y_val = val_stan["sentiment"]

# Train standard model
model_stan = MultinomialNB()
model_stan.fit(xstanfit, y_train)

# Train Moses model
model_moses = MultinomialNB()
model_moses.fit(xmosesfit, train_moses["sentiment"])


results = model_stan.predict(xstantest)
print(metrics.accuracy_score(test_stan['sentiment'], results))

# Predict validation data
results = model_stan.predict(xstanval)

# Calculate confusion matrix
cm = confusion_matrix(val_stan["sentiment"], results)

# Plot
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("TF-IDF Naive Bayes Confusion Matrix")
plt.show()
