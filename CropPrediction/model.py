import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Encode categorical labels into numerical values
c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes

# Prepare features and labels
y = df.target
X = df[['N','P','K','temperature','humidity','ph','rainfall']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVC model with a polynomial kernel
svc_poly = SVC(kernel='poly', degree=3)  # You can adjust the degree as needed
svc_poly.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = svc_poly.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

def predict_crop(input_features):
    # Scale the input features using the same scaler used for training
    input_features_scaled = scaler.transform([input_features])

    # Predict the class
    prediction = svc_poly.predict(input_features_scaled)
    predicted_class = targets[prediction[0]]

    return predicted_class

# Example input from user, one by one
features = ['Nitrogen Contain in Soil', 'Phophoras contain in Soil', 'Potasssium contain in Soil', 'temperature in ℃', 'humidity of Air', 'ph of Soil', 'rainfall in cm']
user_input = []

print("Please enter the following features one by one:")

for feature in features:
    # For each feature, prompt the user to enter its value
    print(f"Enter {feature}:")
    value = float(input())  # Convert the input directly to float
    user_input.append(value)

# Predict and print the crop recommendation
predicted_crop = predict_crop(user_input)
print(f"Recommended Crop: {predicted_crop}")

import joblib

# Save the model
joblib.dump(svc_poly, 'svc_poly_model.pkl')

# Save the scaler
joblib.dump(scaler, 'min_max_scaler.pkl')
import joblib

# Save the targets dictionary
joblib.dump(targets, 'targets.pkl')


