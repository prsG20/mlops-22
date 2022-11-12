from flask import Flask
from flask import request
from joblib import load

import sys
sys.path.append('.')
import os

from utils import preprocess_digits

from sklearn import datasets
digits = datasets.load_digits()
data, label = preprocess_digits(digits)




app = Flask(__name__)

@app.route('/')
def hello():
    return "<b> Hello world </b>"

@app.route('/sum', methods = ['POST'])
def sum(x,y):
    print(request.json)
    x = request.json['x']
    y = request.json['y']
    z = x+y
    return {'sum': z}

model_path = "/Users/prashantgautam/mlops-22/svm_gamma=0.001_C=0.7.joblib"
model = load(model_path)

@app.route('/predict', methods = ['POST'])
def predict_digit():
    i = request.json['i']
    image = data[i]
    
    predicted = model.predict([image])
    return  {"prediction": int(predicted[0])}

#export FLASK_APP=api/app.py
#flask run
#curl -X POST 'https://127.0.0.1:5000/sum' -H 'Content-Type: application/json' -d '{"x":5, "y":4}'

#curl -X POST 'https://127.0.0.1:5000/predict' -H 'Content-Type: application/json' -d '{"i":5}'