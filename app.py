from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('trained_model/model.pkl', 'rb'))
scaler = pickle.load(open('trained_model/scaler.pkl', 'rb'))

app = Flask(__name__)
CORS(app,origins=["https://sriganesh-73.github.io"])
# Serve index.html
@app.route('/')
def testing():
    return "Flask Api testing"

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  
    input_data = np.array(data['features']).reshape(1, -1)  
    input_data = scaler.transform(input_data)  
    
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5500,debug=True)
