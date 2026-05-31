import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Page config
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩"
)

st.title("📩 SMS Spam Classifier")

message = st.text_area("Enter your message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        # Convert text to TF-IDF features
        vector = tfidf.transform([message])

        # Predict
        prediction = model.predict(vector)[0]

        # Display result
        if prediction == "spam":
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

        # Show confidence score
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]

            spam_index = list(model.classes_).index("spam")
            spam_prob = probs[spam_index]

            st.write(f"Spam Probability: {spam_prob*100:.2f}%")
            st.progress(float(spam_prob))