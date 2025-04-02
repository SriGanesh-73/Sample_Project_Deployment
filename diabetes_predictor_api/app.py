from flask import Flask, request, jsonify
import pickle
import numpy as np

#load the model and scaler

model = pickle.load(open('../trained_model/model.pkl','rb'))
scaler = pickle.load(open('../trained_model/scaler.pkl','rb'))

app = Flask(__name__)

@app.route('/predict',methods = ['POST'])
def predict():
    data = request.json #receive json input
    input_data = np.array(data['features']).reshape(1,-1) #converting into numpy array
    input_data = scaler.transform(input_data) #standardizing input_data
    
    prediction = model.predict(input_data)
    return jsonify({'prediction':int(prediction[0])})

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5500,debug=True)