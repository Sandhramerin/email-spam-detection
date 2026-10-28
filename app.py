
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Email Spam Detection", page_icon="📧")

st.title("📧 Email Spam Detection")
st.write("A Machine Learning project to detect spam emails.")

# Load the trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

st.subheader("Enter Email Features")

st.info(
    "This first version uses the numerical features from the "
    "Spambase dataset. It does not accept raw email text yet."
)

st.write("Enter the 57 numerical features below.")

features = []

for i in range(57):
    value = st.number_input(
        f"Feature {i + 1}",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key=f"feature_{i}"
    )
    features.append(value)

if st.button("Check Email"):
    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 Prediction: SPAM")
    else:
        st.success("✅ Prediction: NOT SPAM")
