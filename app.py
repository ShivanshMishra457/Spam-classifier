"""
Streamlit Web App — Spam Email Classifier
Run with: streamlit run app.py
"""

import streamlit as st
from spam_classifier import train, predict


@st.cache_resource
def get_model():
    return train()


st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Spam Email Classifier")
st.caption("Naive Bayes + TF-IDF model — built by Shivansh Mishra")

model, vectorizer = get_model()

st.divider()

SAMPLES = {
    "🏆 Prize winner (spam)": "Congratulations! You've won a FREE iPhone. Click now to claim your prize immediately!",
    "🚨 Urgent threat (spam)": "URGENT: Your bank account has been suspended. Verify your password immediately.",
    "💰 Free cash (spam)": "Earn $5000 per week from home! Guaranteed income. Click here now for your free cash.",
    "📅 Meeting invite (ham)": "Hi team, are we still meeting at 3pm today? Please confirm. Thanks.",
    "📋 Project update (ham)": "Hey, just wanted to share the project update. I pushed the fix, please review when you can.",
    "🍕 Lunch plan (ham)": "Want to grab lunch tomorrow? There's a new place near the office I've been wanting to try.",
}

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

        vec = vectorizer.transform([user_input])
        probs = model.predict_proba(vec)[0]
        col_spam, col_ham = st.columns(2)
        col_spam.metric("Spam probability", f"{probs[1]*100:.1f}%")
        col_ham.metric("Ham probability", f"{probs[0]*100:.1f}%")
    else:
        st.warning("Please enter a message to classify.")

st.divider()
st.subheader("How it works")
c1, c2, c3 = st.columns(3)
c1.info("**1. Preprocess**\nClean text, remove stopwords")
c2.info("**2. TF-IDF**\nConvert words to numeric vectors")
c3.info("**3. Naive Bayes**\nCalculate P(spam | words)")

with st.expander("About the model"):
    st.markdown("""
- **Algorithm**: Multinomial Naive Bayes  
- **Features**: TF-IDF vectors (top 5,000 words)  
- **Training data**: 320 labeled spam/ham messages  
- **Test accuracy**: ~98%  
- **Split**: 80% train / 20% test  
""")
