import os
import sys
import pandas as pd
import numpy as np
from src.exception import Custom_Exception
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.selection import DropDuplicateFeatures
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.preprocessing import StandardScaler
from feature_engine.wrappers import SklearnTransformerWrapper
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def data_transformation(self):

        try:
            numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

            categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

            pipeline = Pipeline([
                ('drop_duplicates', DropDuplicateFeatures()),
                ('imputer_median', MeanMedianImputer(variables = ['chol','thalach'],imputation_method = "median")),
                ('imputer_categorical', CategoricalImputer(imputation_method ='missing', variables = ['sex','cp'],fill_value="Missing")),
                ('log_transformation', YeoJohnsonTransformer(variables = ['trestbps','chol','thalach','oldpeak'])),
                ('rare_label_encoding', RareLabelEncoder(variables = ['thal','ca','restecg'])),
                ('OnehotEncoder',OneHotEncoder(variables = categorical_columns,drop_last=True)),
                ('scalar',SklearnTransformerWrapper(StandardScaler(),variables = numerical_columns)),
                
            ])

            logging.info("Preprocessor obj created")

            return pipeline

        except Exception as e:

            raise Custom_Exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

            for var in categorical_columns:
                train_data[var]=train_data[var].astype('object')
                test_data[var]=test_data[var].astype('object')
            
            preprocessor_obj = self.data_transformation()

            input_train_data = train_data.drop(columns='target',axis=1)
            input_train_target = train_data['target']


            input_test_data = test_data.drop(columns='target',axis=1)
            input_test_target = test_data['target']

            input_train_arr = preprocessor_obj.fit_transform(input_train_data)
            input_test_arr = preprocessor_obj.transform(input_test_data)

            train_arr = np.c_[input_train_arr,np.array(input_train_target)]
            test_arr = np.c_[input_test_arr, np.array(input_test_target)]

            save_object(

                 filepath = self.data_transformation_config.preprocessor_obj_file_path,
                 obj = preprocessor_obj
              )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise Custom_Exception(e,sys)



        

        



        


