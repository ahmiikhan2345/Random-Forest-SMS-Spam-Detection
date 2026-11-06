
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Load trained model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Website heading
st.title("📩 SMS Spam Detection")
st.write("Random Forest Machine Learning Model")

st.markdown("---")

# User input
message = st.text_area(
    "Enter your SMS message:",
    placeholder="Type your message here...",
    height=150
)

# Prediction
if st.button("Predict Message", type="primary"):
    if message.strip():
        text_data = vectorizer.transform([message])
        prediction = model.predict(text_data)[0]

        if prediction == "spam":
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ This is a Safe (Ham) Message!")

    else:
        st.warning("Please enter a message first.")

st.markdown("---")
st.caption("Built using Python, Random Forest, TF-IDF & Streamlit")
