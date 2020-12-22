import numpy as np
import pandas as pd
import os
from typing import Dict, List
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier as KNN
from deploy.constants.request_parameters import RequestParameters


class Iris(object):

    def __init__(self):
        self.knn_model = joblib.load('iris_model1.pkl')
        
    def predict(self, request: Dict, features_names=None) -> Dict:
        VAR1 = request[RequestParameters.VAR1]
        VAR2 = request[RequestParameters.VAR2]
        VAR3 = request[RequestParameters.VAR3]
        VAR4 = request[RequestParameters.VAR4]
        
        x_score = np.array([VAR1,VAR2,VAR3,VAR4])
        
        output1 = self.knn_model.predict(x_score)

        return self.make_response(x_score,output1)

    def make_response(self, input1: List, output1: List) -> Dict:
        response = []
        item = {
                "Input": input1, "Class": output1}
        response.append(item)

        return response
