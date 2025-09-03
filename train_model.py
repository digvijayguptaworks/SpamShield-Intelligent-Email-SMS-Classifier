import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# -------------------------
# Preprocessing Function
# -------------------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # lowercase
    text = ''.join([ch for ch in text if ch not in string.punctuation])  # remove punctuation
    words = text.split()
    words = [word for word in words if word.isalnum()]  # keep only words/numbers
    words = [word for word in words if word not in stopwords.words('english')]  # remove stopwords
    words = [ps.stem(word) for word in words]  # stemming
    return " ".join(words)

# 1. Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")[["v1", "v2"]]
df = df.rename(columns={"v1": "label", "v2": "message"})

# 2. Encode labels
df["label_num"] = df.label.map({"ham": 0, "spam": 1})

# 3. Apply preprocessing
df["transformed"] = df["message"].apply(transform_text)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["transformed"], df["label_num"], test_size=0.2, random_state=42
)

# 5. Vectorize text
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 8. Save vectorizer + model
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
