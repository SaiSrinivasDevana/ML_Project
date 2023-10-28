import os
import sys
from src.logger import logging
from src.exception import Custom_Exception
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass
class  DataIngestionConfig:
    train_data_path : str=os.path.join("artifacts","train.csv")
    test_data_path : str=os.path.join("artifacts","test.csv")
    raw_data_path : str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def data_ingestion(self):
        
        logging.info("Data Ingestion Process has started")

        try:
            data=pd.read_csv("notebook\data\heart_disease_dataset.csv")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split Started")

            train_data,test_data=train_test_split(data,test_size=0.2,random_state=323,shuffle=True)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as E:
            raise Custom_Exception(E,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.data_ingestion()

        
        