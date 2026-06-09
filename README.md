# 📧 Spam Email Classifier

A machine learning project that classifies emails and SMS messages as **spam** or **ham (legitimate)** using the Naive Bayes algorithm with TF-IDF feature extraction.

Built as part of an AI/ML internship project.

---

## Demo

Run the interactive web app locally with Streamlit — enter any message and get an instant prediction with confidence scores.

---

## Features

- Trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset (5,572 messages)
- TF-IDF vectorization for text feature extraction
- Multinomial Naive Bayes classifier (~98% test accuracy)
- Streamlit web UI for interactive testing
- CLI mode for quick predictions in the terminal
- Model persistence with `joblib` — train once, reuse anytime

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data handling | `pandas` |
| ML model | `scikit-learn` |
| Text features | `TfidfVectorizer` |
| Web UI | `streamlit` |
| Model saving | `joblib` |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the web app

```bash
streamlit run app.py
```

The dataset downloads and the model trains automatically on the first run. Opens in your browser at `http://localhost:8501`.

### 4. Or run the CLI version

```bash
python spam_classifier.py
```

---

## How It Works

```
Raw text
   ↓
Preprocessing (lowercase, remove stopwords)
   ↓
TF-IDF Vectorization (top 5,000 word features)
   ↓
Multinomial Naive Bayes
   ↓
Prediction: SPAM or HAM + confidence score
```

**Naive Bayes** works by calculating the probability that each word appears in spam vs. ham emails. It then multiplies these probabilities together (Bayes' theorem) to give the final verdict.

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts raw text into numeric vectors, giving higher weight to words that are distinctive to a message and lower weight to common words.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~98% |
| Precision (spam) | ~97% |
| Recall (spam) | ~95% |
| F1 Score (spam) | ~96% |

---

## Sample Predictions

| Message | Prediction |
|---------|-----------|
| "Congratulations! You've won a FREE iPhone. Click now!" | 🚨 SPAM |
| "Hey, are we still meeting at 3pm today?" | ✅ HAM |
| "URGENT: Verify your bank account immediately." | 🚨 SPAM |
| "Thanks for sending the report, I'll review it tonight." | ✅ HAM |

---

## Project Structure

```
spam-classifier/
├── spam_classifier.py   # Core ML model: training, evaluation, prediction
├── app.py               # Streamlit web app
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## What I Learned

- Text preprocessing and feature extraction with TF-IDF
- How Naive Bayes works and why it suits NLP classification
- Evaluating classifiers using precision, recall, and F1 score
- Building and deploying an interactive ML app with Streamlit

---

## Dataset

[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
5,572 labeled SMS messages (4,825 ham, 747 spam)

---

## License

MIT
