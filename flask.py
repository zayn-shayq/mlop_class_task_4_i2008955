from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Welcome to the Accident Prediction Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the input data is in the correct format
    data = request.get_json(force=True)
    # Assuming the data is a list of features for prediction
    features = np.array(data['features']).reshape(1, -1)
    
    # Generate predictions
    prediction = model.predict(features)
    
    # You can add post-processing to the prediction here if needed
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
