"""
Spam Email Classifier
Author: Shivansh
Description: Naive Bayes classifier trained on built-in spam/ham dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_dataset():
    """Return a built-in labeled dataset of spam and ham messages."""
    spam = [
        "Congratulations you won a free iPhone click now to claim your prize",
        "URGENT claim your cash prize immediately free reward winner selected",
        "You have been selected as our lucky winner claim your million dollars now",
        "Free entry win guaranteed cash prize click here now limited offer",
        "Your bank account has been suspended verify your password immediately urgent",
        "Win a brand new car click to claim your reward today free gift",
        "You are our lucky winner get free gift card now click immediately",
        "Limited offer buy cheap pills discount pharmacy online free delivery",
        "Earn money from home guaranteed income free sign up no experience needed",
        "Nigerian prince inheritance claim your share of million dollars urgent",
        "Congratulations prize winner call now to claim your reward free cash",
        "Free iPhone winner selected click immediately urgent limited time offer",
        "Amazing deal cheap discount buy now limited time offer free shipping",
        "Earn unlimited income work from home free registration click now",
        "Claim free gift voucher winner selected congratulations act now",
        "Free loan approved claim now no credit check required urgent reply",
        "Win lottery jackpot selected lucky winner claim prize now click here",
        "You won cash prize free entry selected winner claim now urgent",
        "Congratulations you have won a luxury holiday click to claim free",
        "Your credit card has been compromised call this number immediately urgent",
        "Special offer free subscription click now to activate your reward",
        "Urgent message your account will be closed verify details immediately",
        "You have been chosen click here to receive your free cash reward now",
        "Double your income from home guaranteed results free sign up today",
        "Exclusive deal only for you free gift card click to claim prize now",
        "Winner announcement you have won free holiday package call immediately",
        "Cheap viagra cialis pills online pharmacy discount free delivery now",
        "Make money fast guaranteed income work from home click here free",
        "Your parcel is waiting claim it free click the link now urgent",
        "Free bitcoin reward claim now limited time offer winner selected urgent",
        "Alert your payment failed update your details immediately to avoid suspension",
        "Congratulations selected for free trial click now to activate reward",
        "Urgent your subscription expires today renew now free bonus included",
        "You qualify for a free loan apply now no credit check guaranteed",
        "Win big cash prizes enter free now guaranteed winner selected today",
        "Free mobile recharge click now winner selected congratulations urgent",
        "Your account has suspicious activity verify now to avoid being locked",
        "Claim your free amazon gift card winner selected click immediately now",
        "Exclusive offer for you free insurance quote click now to save money",
        "Final notice your debt can be cleared free consultation click now",
    ] * 4

    ham = [
        "Hey are we still meeting at 3pm today please let me know",
        "Thanks for sending the report I will review it tonight",
        "Hi team the project update is ready please review the attached document",
        "Can we reschedule the call to tomorrow afternoon works better for me",
        "Just wanted to check if you received my last email please confirm",
        "The bug fix has been deployed please test when you get a chance",
        "Lunch tomorrow sounds good what time works for you let me know",
        "Please find attached the document you requested for the meeting",
        "Quick question about the deadline can we discuss this today",
        "Great work on the presentation everyone was really impressed",
        "I will be a few minutes late to the meeting sorry for the delay",
        "The client approved the proposal we can start next week",
        "Can you share the notes from yesterday meeting thanks so much",
        "Happy to help let me know if you need anything else from me",
        "The server is back online everything looks good now no issues",
        "See you at the office tomorrow have a good evening",
        "I have reviewed the code changes looks good to merge well done",
        "Thanks for the quick response really appreciate your help",
        "The report is due Friday please send your section by Thursday morning",
        "Coffee catch up this week free anytime Wednesday afternoon let me know",
        "Just a reminder that the team standup is at 10am tomorrow morning",
        "Could you send me the invoice details for last month please thanks",
        "The new feature is working great users are really happy with it",
        "I will be out of office next Monday back on Tuesday morning",
        "Please review the pull request when you have a moment thanks",
        "The meeting has been moved to 4pm today hope that works for you",
        "Can you help me with the database query I am a bit stuck on it",
        "Just checking in how is the project going any blockers at the moment",
        "The design mockups are ready for review let me know your thoughts",
        "Happy birthday hope you have a wonderful day enjoy your celebrations",
        "The quarterly report is attached please review before the board meeting",
        "I shared the document with you please check your google drive",
        "Running five minutes late to the call sorry be there shortly",
        "The test results look good we are ready to push to production",
        "Thank you for attending the workshop today hope it was helpful",
        "Could we schedule a quick call this week to discuss the proposal",
        "The package arrived safely thank you for sending it so quickly",
        "I have updated the spreadsheet with the latest numbers please check",
        "Looking forward to the team lunch on Friday should be fun",
        "Please confirm your attendance for the conference by end of day",
    ] * 4

    texts = spam + ham
    labels = [1] * len(spam) + [0] * len(ham)
    return pd.DataFrame({"text": texts, "label": labels})


def train(df=None):
    """Train the Naive Bayes classifier and return model + vectorizer."""
    if df is None:
        df = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("--- Model Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, vectorizer


def predict(text, model, vectorizer):
    """Predict whether a given text is spam or ham."""
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[pred] * 100
    return label, confidence


def interactive_mode(model, vectorizer):
    """Run an interactive CLI loop for testing the classifier."""
    print("\n--- Interactive Mode ---")
    print("Type an email/SMS message to classify it. Type 'quit' to exit.\n")
    while True:
        text = input("Enter message: ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue
        label, confidence = predict(text, model, vectorizer)
        print(f"  → Prediction : {label}  ({confidence:.1f}% confidence)\n")


if __name__ == "__main__":
    model, vectorizer = train()

    samples = [
        "Congratulations! You've won a FREE iPhone. Click now to claim your prize!",
        "Hey, are we still meeting at 3pm today?",
        "URGENT: Your bank account has been suspended. Verify immediately.",
        "Thanks for sending the report, I'll review it tonight.",
    ]

    print("\n--- Sample Predictions ---")
    for s in samples:
        label, conf = predict(s, model, vectorizer)
        print(f"  [{label:4s}] ({conf:.1f}%)  {s[:60]}...")

    interactive_mode(model, vectorizer)
