# 📩 SpamShield: SMS Spam Classifier

A machine learning web app that classifies a text message as **spam** or **ham** (legitimate) using Natural Language Processing and a Naive Bayes classifier. Paste a message, click Predict, and get an instant verdict.

![App screenshot](screenshots/app.png)

## Features

- Instant spam / safe prediction through a clean dark-themed web UI (Streamlit)
- Text preprocessing pipeline: lowercasing, punctuation removal, stopword removal and stemming
- TF-IDF features with a Multinomial Naive Bayes model
- Pre-trained model included, so the app runs immediately
- Training script and notebook included, so you can retrain or experiment

## How it works

1. **Clean the text:** lowercase it, remove punctuation, drop common English stopwords (like "the", "is") and reduce each word to its root with the Porter stemmer (for example "winning" becomes "win").
2. **Convert to numbers:** TF-IDF turns each message into a vector over the 3,000 most important words.
3. **Classify:** a Multinomial Naive Bayes model predicts spam (1) or ham (0).

## Model performance

Trained on a labelled SMS dataset (`spam.csv`) with an 80/20 train/test split (`random_state=42`).

| Metric | Score |
|---|---|
| Accuracy | XX.XX% |
| Precision (spam) | XX.XX% |
| Recall (spam) | XX.XX% |

> Spam datasets are imbalanced, with far more ham than spam messages. That's why precision and recall on the spam class are reported along with accuracy.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| NLP | NLTK (stopwords, Porter stemmer) |
| Machine learning |
