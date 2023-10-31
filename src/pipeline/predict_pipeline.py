import os
import sys
from src.exception import Custom_Exception
from src.utils import load_object
import pandas as pd

class Predict_Pipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model= load_object(model_path)
            print("After Loading")
            preprocessor = load_object(preprocessor_path)
            transformed_features = preprocessor.transform(features)
            prediction = model.predict(transformed_features)
            return prediction
        
        except Exception as e:
            raise Custom_Exception(e,sys)

class Custom_Data:
    
    def __init__(self,
                 age : int,
                 sex : str,
                 cp : str,
                 trestbps : int,
                 chol : int,
                 fbs : str,
                 restecg : str,
                 thalach : int,
                 exang : str,
                 oldpeak : int,
                 slope : str,
                 ca : str,
                 thal : str):
        self.age = age
        self. sex = sex,
        self.cp = cp,
        self.trestbps = trestbps,
        self.chol = chol,
        self.fbs = fbs,
        self.restecg = restecg,
        self.thalach = thalach,
        self.exang = exang,
        self.oldpeak = oldpeak,
        self.slope = slope,
        self.ca = ca,
        self.thal =thal
    
    def get_custom_data(self):
        try:
            custom_input_data = {
                "age" : self.age,
                "sex" : self.sex,
                "cp" : self.cp,
                "trestbps" : self.trestbps,
                "chol" : self.chol,
                "fbs" : self.fbs,
                "restecg" : self.restecg,
                "thalach" : self.thalach,
                "exang" : self.exang,
                "oldpeak" : self.oldpeak,
                "slope" : self.slope,
                "ca" : self.ca,
                "thal" : self.thal
            }
            return pd.DataFrame(custom_input_data)

        except Exception as e:
            raise Custom_Exception

        