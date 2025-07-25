from flask import Flask, render_template, request, jsonify
import pickle
import easyocr
import cv2
import pytesseract
import os
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

app = Flask(__name__)

# Load dataset
data_path = 'new_data.csv'
df = pd.read_csv(data_path)

# Inspect the data
print("Dataset loaded. First few rows:\n", df.head())

# Assuming the dataset has 'text' and 'label' columns
# Preprocessing function for text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

df['tweet'] = df['tweet'].apply(preprocess_text)

# Load the cyberbullying detection model
ytb_model = open("Cyberbull.pkl", "rb")
new_model = pickle.load(ytb_model)

def predict_cyberbullying(text):
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    input_tfidf = vectorizer.transform([preprocess_text(text)])
    prediction = model.predict(input_tfidf)    
    return "Cyberbullying" if prediction[0] == 1 else "Not Cyberbullying"

def extract_text_easyocr(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    total_text = " ".join([text for (_, text, _) in result])
    return total_text

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    text = ""
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # Process one frame per second
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_text = pytesseract.image_to_string(gray_frame)
            text += " " + frame_text

    cap.release()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    input_text = request.form['text']
    prediction = predict_cyberbullying(input_text)
    return jsonify({'result': prediction})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    image_file = request.files['image']
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    extracted_text = extract_text_easyocr(image_path)
    prediction = predict_cyberbullying(extracted_text)
    os.remove(image_path)
    return jsonify({'extracted_text': extracted_text, 'result': prediction})

@app.route('/predict_video', methods=['POST'])
def predict_video():
    video_file = request.files['video']
    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)

    extracted_text = extract_text_from_video(video_path)
    prediction = predict_cyberbullying(extracted_text)
    os.remove(video_path)
    return jsonify({'extracted_text': extracted_text, 'result': prediction})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
