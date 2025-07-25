import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("new_data.csv", encoding="ISO-8859-1")  # or "Windows-1252"
#data_path = 'new_data.csv'
#df = pd.read_csv(data_path)

# Inspect the data
print("Dataset loaded. First few rows:\n", df.head())

# Assuming the dataset has 'text' and 'label' columns
# Preprocessing function for text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

df['tweet'] = df['tweet'].apply(preprocess_text)

# Split dataset
X = df['tweet']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train multiple models and evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model and vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print(f"Best model saved: {type(best_model).__name__} with accuracy {best_accuracy}")

# Prediction function
def predict_text(input_text):
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    input_tfidf = vectorizer.transform([preprocess_text(input_text)])
    prediction = model.predict(input_tfidf)
    return prediction[0]

# Example usage
example_text = " Get fucking real dude."
print("Prediction for input text:", predict_text(example_text))
