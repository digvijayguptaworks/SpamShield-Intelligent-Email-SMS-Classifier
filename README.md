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
| Machine learning | scikit-learn (TF-IDF, Multinomial Naive Bayes) |
| Data | pandas |
| Web app | Streamlit |

## Project structure

```
├── app.py                    # Streamlit web app
├── train_model.py            # Trains the model and saves the .pkl files
├── sms_spam_detection.ipynb  # Exploration and experiments
├── model.pkl                 # Trained Naive Bayes model
├── vectorizer.pkl            # Fitted TF-IDF vectorizer
├── spam.csv                  # Training dataset
├── requirements.txt          # Python dependencies
└── setup.sh                  # One-step setup script
```

## Getting started

```bash
# 1. Clone the repo
git clone https://github.com/digvijayguptaworks/SpamShield-Intelligent-Email-SMS-Classifier.git
cd SpamShield-Intelligent-Email-SMS-Classifier

# 2. Install dependencies and download NLTK data
./setup.sh
# or manually:
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Retrain the model (optional)

```bash
python train_model.py
```
This prints the test accuracy and regenerates `model.pkl` and `vectorizer.pkl`.

## Example

| Message | Prediction |
|---|---|
| "Congratulations! You've won a free iPhone. Click here to claim now." | 🚨 Spam |
| "Hey, are we still meeting for lunch at 1?" | ✅ Safe |

## Limitations and future work

- The model is trained on **SMS messages only**. It may work on short emails, but this hasn't been tested on real email data.
- It uses a bag-of-words approach, so it doesn't understand context or word order. Newer spam tricks (obfuscated links, unusual spellings) can slip through.
- Planned improvements: a larger multi-source dataset, comparison with other models (Logistic Regression, SVM, a transformer such as BERT), and an API endpoint.

## Author

**Digvijay Gupta**
GitHub: [digvijayguptaworks](https://github.com/digvijayguptaworks)
Email: digvijaygworks@gmail.com
