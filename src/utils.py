import os
import pickle
from src.exception import Custom_Exception
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

def save_object(filepath,obj):

    try:
        dir_name = os.path.dirname(filepath)

        os.makedirs(dir_name,exist_ok = True)

        with open(filepath,"wb") as fileobj:
            pickle.dump(obj,fileobj)
    
    except Exception as e:
        raise Custom_Exception(e,sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = recall_score(y_train, y_train_pred)

            test_model_score = recall_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise Custom_Exception(e,sys)
    
    
