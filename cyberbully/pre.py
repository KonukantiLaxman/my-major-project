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
df = pd.read_csv("new_data.csv", encoding="ISO-8859-1")
 

# Inspect the data
print("Dataset loaded. First few rows:\n", df.head())

# Assuming the dataset has 'text' and 'label' columns
# Preprocessing function for text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

df['tweet'] = df['tweet'].apply(preprocess_text)

 

# Prediction function
def predict_text(input_text):
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    input_tfidf = vectorizer.transform([preprocess_text(input_text)])
    prediction = model.predict(input_tfidf)
    if prediction[0] == 0:
        return "The content was cyberbully free."
    else:
        return "The content was containing cyberbullying."

