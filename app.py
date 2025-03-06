from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# List of symptoms (Make sure this matches your dataset)
SYMPTOMS = ["fever", "cough", "headache", "fatigue", "nausea", "sore throat", "chills"]

@app.route("/")
def home():
    return render_template("index.html", symptoms=SYMPTOMS)

@app.route("/predict", methods=["POST"])
def predict():
    selected_symptoms = request.form.getlist("symptoms")
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]

    # Make prediction
    prediction_encoded = model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]

    return render_template("result.html", disease=predicted_disease)

if __name__ == "__main__":
    app.run(debug=True)
