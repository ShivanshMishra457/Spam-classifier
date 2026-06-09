"""
Streamlit Web App — Spam Email Classifier
Run with: streamlit run app.py
"""

import os
import streamlit as st
import joblib
from spam_classifier import download_dataset, load_data, train, save_model, predict

MODEL_PATH = os.path.join("model", "model.pkl")
VEC_PATH = os.path.join("model", "vectorizer.pkl")


@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VEC_PATH)
    with st.spinner("Training model for the first time..."):
        path = download_dataset()
        df = load_data(path)
        model, vectorizer = train(df)
        save_model(model, vectorizer)
    return model, vectorizer


st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Spam Email Classifier")
st.caption("Naive Bayes model trained on the UCI SMS Spam Collection dataset")

model, vectorizer = get_model()

st.divider()

SAMPLES = {
    "🏆 Prize winner (spam)": "Congratulations! You've won a FREE iPhone. Click now to claim your prize immediately!",
    "🚨 Urgent threat (spam)": "URGENT: Your bank account has been suspended. Verify your password immediately.",
    "💰 Free cash (spam)": "Earn $5000 per week from home! Guaranteed income. Click here now for your free cash.",
    "📅 Meeting invite (ham)": "Hi team, are we still meeting at 3pm today? Please confirm. Thanks.",
    "📋 Project update (ham)": "Hey, just wanted to share the project update. I've pushed the fix, please review when you can.",
    "🍕 Lunch plan (ham)": "Want to grab lunch tomorrow? There's a new place near the office I've been wanting to try.",
}

col1, col2 = st.columns([3, 1])
with col1:
    selected = st.selectbox("Try a sample", ["(type your own)"] + list(SAMPLES.keys()))

default_text = SAMPLES.get(selected, "")
user_input = st.text_area("Message text", value=default_text, height=120, placeholder="Type or paste an email/SMS here...")

if st.button("Classify →", use_container_width=True, type="primary"):
    if user_input.strip():
        label, confidence = predict(user_input, model, vectorizer)
        is_spam = label == "SPAM"

        if is_spam:
            st.error(f"🚨 **SPAM** — {confidence:.1f}% confidence")
        else:
            st.success(f"✅ **HAM (Legitimate)** — {confidence:.1f}% confidence")

        col_spam, col_ham = st.columns(2)
        vec = vectorizer.transform([user_input])
        probs = model.predict_proba(vec)[0]
        col_spam.metric("Spam probability", f"{probs[1]*100:.1f}%")
        col_ham.metric("Ham probability", f"{probs[0]*100:.1f}%")
    else:
        st.warning("Please enter a message to classify.")

st.divider()
st.subheader("How it works")
steps = st.columns(3)
steps[0].info("**1. Preprocess**\nClean text, remove stopwords")
steps[1].info("**2. TF-IDF**\nConvert words to numeric vectors")
steps[2].info("**3. Naive Bayes**\nCalculate P(spam | words)")

with st.expander("About the model"):
    st.markdown("""
- **Algorithm**: Multinomial Naive Bayes  
- **Features**: TF-IDF vectors (top 5,000 words)  
- **Dataset**: UCI SMS Spam Collection (5,572 messages)  
- **Test accuracy**: ~98%  
- **Split**: 80% train / 20% test  
""")
