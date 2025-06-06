# Import Required Libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify, render_template

# Load Dataset
#df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df = pd.read_csv(r"C:\Users\Swetha Sridharan\Desktop\SPAM DETECTION\spam.csv", encoding='latin-1')[['v1', 'v2']]

df.columns = ['label', 'message']

# Convert Labels: 'ham' → 0 (Not Spam), 'spam' → 1 (Spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# Split Dataset into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_message'], df['label'], test_size=0.2, random_state=42)

# Convert Text to Numerical Vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate Model Performance
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to Predict New Messages
def predict_spam(text):
    text = clean_text(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Flask Web App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['message']
    prediction = predict_spam(data)
    return render_template('index.html', prediction_text=f'Message Classification: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
