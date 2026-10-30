
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("email_text_model.pkl")

model = load_model()

# App title
st.title("📧 Email Spam Detection")
st.write("Check whether an email is Spam or Ham using Machine Learning.")

st.divider()

# Email input
st.subheader("Enter your email")

subject = st.text_input(
    "Email Subject",
    placeholder="Enter the email subject here"
)

message = st.text_area(
    "Email Message",
    placeholder="Paste your email message here...",
    height=200
)

# Prediction button
if st.button("Check Email", type="primary"):

    email_text = subject + " " + message

    if not email_text.strip():
        st.warning("Please enter an email subject or message.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        confidence = max(probabilities) * 100

        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM (HAM)")

        st.write(f"Model confidence: {confidence:.2f}%")

st.divider()

st.caption("Powered by Python, TF-IDF and Logistic Regression.")
