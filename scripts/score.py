# -*- coding: utf-8 -*-
"""
Created 

@author: 
"""

import json
import numpy
import joblib


# Load Model
def init():
    global model
    model_path = "C:/Users/rodri/Downloads/convert_ml_experiments_code_prod/models/sklearn_regression_model.pkl"
    model = joblib.load(model_path)
    

# Prepare Data and Score Data
def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    return {"result": result.tolist()}


init()


test_row = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'
request_header = {}
prediction = run(test_row, {})
print("Test result: ", prediction)
