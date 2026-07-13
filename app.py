from flask import Flask, render_template, request
import joblib

from utils.text_preprocessor import transform_text

import os

# Flask App
app = Flask(__name__)

# Load Saved Model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # User input
    message = request.form["message"]

    # Preprocess text
    transformed_message = transform_text(message)

    # Convert into TF-IDF vector
    vector_input = vectorizer.transform([transformed_message])

    # Predict
    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        result = "🚨 Spam Message"
        color = "red"
    else:
        result = "✅ Safe Message"
        color = "green"

    return render_template(
        "index.html",
        prediction=result,
        color=color,
        message=message
)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )