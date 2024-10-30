from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('svc_poly_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('min_max_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the target dictionary
with open('targets.pkl', 'rb') as targets_file:
    targets = pickle.load(targets_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    user_input = [float(request.form[feature]) for feature in features]

    # Scale input features
    input_features_scaled = scaler.transform([user_input])

    # Predict the crop
    prediction = predict_crop(input_features_scaled)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction)

def predict_crop(input_features_scaled):
    # Predict the class
    prediction = model.predict(input_features_scaled)
    predicted_class = targets[prediction[0]]

    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
