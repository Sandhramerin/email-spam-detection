
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the email dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

data = pd.read_csv(url, header=None)

# Separate email features and spam labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the machine-learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("Model training successful!")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
import pickle

with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as spam_model.pkl")
