import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import pandas as pd
import re

# Load the dataset
df = pd.read_csv("new_data.csv", encoding="ISO-8859-1")
 

# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

df["tweet"] = df["tweet"].astype(str).apply(clean_text)

# Parameters
max_words = 10000  # Increase vocabulary size
max_len = 150  # Increase sequence length

# Tokenization
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["tweet"])
sequences = tokenizer.texts_to_sequences(df["tweet"])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="pre", truncating="pre")

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df["label"], test_size=0.2, random_state=42)

# Define the improved LSTM Model
model = Sequential([
    Embedding(max_words, 200, input_length=max_len),  # Increase embedding size
    Bidirectional(LSTM(128, return_sequences=True)),  # Use Bidirectional LSTM
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# Compile the model with a lower learning rate
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=["accuracy"])

# Train the model with more epochs
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("lstm_model_improved.h5")

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Predict on new input
def predict_text(text, tokenizer, model, max_len=150):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="pre", truncating="pre")
    prediction = model.predict(padded)
    return "Positive" if prediction[0][0] > 0.5 else "Negative"

# Load and test prediction
model = tf.keras.models.load_model("lstm_model_improved.h5")
result = predict_text("how are you", tokenizer, model)
print("Prediction:", result)
