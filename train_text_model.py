
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the email dataset
data = pd.read_csv("enron_spam_data.csv")

# 2. Clean column names
data.columns = data.columns.str.strip()

# 3. Prepare email text
data["Subject"] = data["Subject"].fillna("")
data["Message"] = data["Message"].fillna("")

data["Email_Text"] = (
    data["Subject"].astype(str) + " " +
    data["Message"].astype(str)
)

# 4. Prepare spam and ham labels
data["Spam/Ham"] = data["Spam/Ham"].astype(str).str.strip().str.lower()

data = data[data["Spam/Ham"].isin(["spam", "ham"])]

X = data["Email_Text"]
y = data["Spam/Ham"]

# 5. Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Total emails:", len(data))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 9. Save the trained model
joblib.dump(model, "email_text_model.pkl")

print("\nModel saved as email_text_model.pkl")
