import os
import sys
from src.exception import Custom_Exception
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from src.utils import evaluate_models,save_object
from sklearn.metrics import recall_score

@dataclass
class ModelTrainerConfig:
    model_train_obj_file =  os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            X_train,y_train = (
              train_arr[:,:-1],train_arr[:,-1])
            X_test,y_test = (
               test_arr[:,:-1],test_arr[:,-1])
            model={

                
                "logistic_regression_model" : LogisticRegression(),
                "random_forest_model" : RandomForestClassifier(),
                "decision_tree_classifier" : DecisionTreeClassifier(),
                "knn_classifier_model" : KNeighborsClassifier(),
                "ada_boost_classifier" : AdaBoostClassifier(),
                "xg_boost_classifier" : XGBClassifier()        

             }

            params={
                
               "logistic_regression_model" : {
                   'class_weight' : ['balanced', None],
                   'penalty' : ['l1','l2',None]

                },
                "random_forest_model" : {
                   'n_estimators' : [100,200,300],
                   'max_features' : ['sqrt','log2',None],
                   'max_depth' : range(1,10,2)
               },
               "decision_tree_classifier" : {
                   
                  'max_depth' : np.arange(1,10,2),
                  'max_leaf_nodes' : np.arange(1,10,2),
                  'class_weight' : ['balanced', None],
                  'criterion' : ['gini', 'entropy']
               },
               "knn_classifier_model" : {
                   'n_neighbors' : np.arange(1,15,2)
               },
               "ada_boost_classifier" : {
                   'learning_rate':[.1,.01,0.2,.001],
                    'n_estimators': [100,200,300]
               },
               "xg_boost_classifier" : {
                   'learning_rate':[0.01,0.1,0.2,0.3,.001],
                   'max_depth' : np.arange(1,10,2),
                   'n_estimators' : [100,200,300],
                   'subsample' : np.arange(0.5,1,0.1)
               }
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=model,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = model[best_model_name]

            


            save_object(
                filepath=self.model_train_config.model_train_obj_file,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            test_scores = recall_score(y_test, predicted)
            return test_scores
        except Exception as e:
            raise Custom_Exception(e,sys)


        